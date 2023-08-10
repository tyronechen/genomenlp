#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# conduct sweep
import argparse
from collections import Counter
import functools
import gzip
import os
from random import shuffle
from warnings import warn
from datasets import ClassLabel, Dataset, DatasetDict, Value, load_dataset, load_metric
import json
import numpy as np
import pandas as pd
import torch
from tokenizers import models, SentencePieceUnigramTokenizer
from tqdm import tqdm
import transformers
from transformers import AutoModelForSequenceClassification, \
    DataCollatorWithPadding, \
    DistilBertConfig, DistilBertForSequenceClassification, HfArgumentParser, \
    LongformerConfig, LongformerForSequenceClassification, \
    PreTrainedTokenizerFast, Trainer, TrainingArguments, set_seed
from transformers.training_args import ParallelMode
from utils import load_args_json, load_args_cmd, get_run_metrics, _cite_me
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import download_data, \
    build_compute_metrics_fn
import wandb

def main():
    parser = argparse.ArgumentParser(
         description='Take HuggingFace dataset and perform parameter sweeping.'
        )
    parser.add_argument('train', type=str,
                        help='path to [ csv | csv.gz | json | parquet ] file')
    parser.add_argument('format', type=str,
                        help='specify input file type [ csv | json | parquet ]')
    parser.add_argument('tokeniser_path', type=str,
                        help='path to tokeniser.json file to load data from')
    parser.add_argument('-t', '--test', type=str, default=None,
                        help='path to [ csv | csv.gz | json | parquet ] file')
    parser.add_argument('-v', '--valid', type=str, default=None,
                        help='path to [ csv | csv.gz | json | parquet ] file')
    parser.add_argument('-m', '--model', type=str, default="distilbert",
                        help='choose model [ distilbert | longformer ] \
                        distilbert handles shorter sequences up to 512 tokens \
                        longformer handles longer sequences up to 4096 tokens \
                        (DEFAULT: distilbert)')
    parser.add_argument('--model_features', type=int, default=None,
                        help='number of features in data to use (DEFAULT: ALL) \
                        NOTE: this is separate from the vocab_size argument. \
                        under normal circumstances (eg a tokeniser generated \
                        by SentencePiece), setting this is not necessary')
    parser.add_argument('-o', '--output_dir', type=str, default="./sweep_out",
                        help='specify path for output (DEFAULT: ./sweep_out)')
    parser.add_argument('-d', '--device', type=str, default="auto",
                        help='choose device [ cpu | cuda:0 ] (DEFAULT: detect)')
    parser.add_argument('-s', '--vocab_size', type=int, default=32000,
                        help='vocabulary size for model configuration')
    parser.add_argument('-w', '--hyperparameter_sweep', type=str, default="",
                        help='run a hyperparameter sweep with config from file')
    parser.add_argument('-l', '--label_names', type=str, default="", nargs="+",
                        help='provide column with label names (DEFAULT: "").')
    parser.add_argument('-n', '--sweep_count', type=int, default=8,
                        help='run n hyperparameter sweeps (DEFAULT: 64)')
    parser.add_argument('-e', '--entity_name', type=str, default="",
                        help='provide wandb team name (if available).')
    parser.add_argument('-p', '--project_name', type=str, default="",
                        help='provide wandb project name (if available).')
    parser.add_argument('-g', '--group_name', type=str, default="sweep",
                        help='provide wandb group name (if desired).')
    parser.add_argument('--partition_percent', type=int, default=100,
                        help='only sweep on a percentage of data (int!). \
                        the percentage will be rounded off depending on \
                        the closest split. for example, 30 will be \
                        rounded to a split of int(100/30)=3, so the \
                        percentage will be ~33. defaults to sweeping \
                        on the full dataset (DEFAULT: 100).')   
    parser.add_argument('--metric_opt', type=str, default="eval/f1",
                        help='score to maximise [ eval/accuracy | \
                        eval/validation | eval/loss | eval/precision | \
                        eval/recall ] (DEFAULT: eval/f1)')
    parser.add_argument('-r', '--resume_sweep', type=str, default=None,
                        help='provide sweep id to resume sweep.')
    parser.add_argument('--fp16_off', action="store_false",
                        help='turn fp16 off for precision / cpu (DEFAULT: ON)')
    parser.add_argument('--wandb_off', action="store_false",
                        help='run hyperparameter tuning using the wandb api \
                        and log training in real time online (DEFAULT: ON)')

    args = parser.parse_args()
    train = args.train
    format = args.format
    model = args.model
    device = args.device
    test = args.test
    valid = args.valid
    tokeniser_path = args.tokeniser_path
    hyperparameter_sweep = args.hyperparameter_sweep
    sweep_count = args.sweep_count
    vocab_size = args.vocab_size
    wandb_state = args.wandb_off
    entity_name = args.entity_name
    project_name = args.project_name
    group_name = args.group_name
    metric_opt = args.metric_opt
    model_features = args.model_features
    output_dir = args.output_dir
    fp16 = args.fp16_off
    resume_sweep = args.resume_sweep
    partition_percent = int(args.partition_percent)
    if wandb_state is True:
        wandb.login()
        args.report_to = "wandb"
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    api = wandb.Api(timeout=10000)
    if device == "gpu" and fp16 == False:
        warn("Training on gpu but fp16 is off. Can enable to increase speed.")
    if device == "cpu" and fp16 == True:
        warn("Training on cpu but fp16 is on. Disabled as incompatible.")
        fp16 = False

    print("\n\nUSING DEVICE:\n", device)
    print("\n\nARGUMENTS:\n", args, "\n\n")

    if os.path.exists(tokeniser_path):
        special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
        print("USING EXISTING TOKENISER:", tokeniser_path)
        tokeniser = PreTrainedTokenizerFast(
            tokenizer_file=tokeniser_path,
            special_tokens=special_tokens,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
            )
        data_collator = DataCollatorWithPadding(tokenizer=tokeniser,
                                                padding="longest")

    infile_paths = dict()
    infile_paths["train"] = train
    if test != None:
        infile_paths["test"] = test
    if valid != None:
        infile_paths["valid"] = valid
    dataset = load_dataset(format, data_files=infile_paths)

    # for frequency based feature selection
    # it is not straightforward to remove vocab from a tokeniser, see below
    # NOTE: https://github.com/huggingface/transformers/issues/15032
    if model_features != None:
        counts = Counter([x for y in dataset['train']['input_ids'] for x in y])
        counts = {
            k: v for k, v in
            sorted(counts.items(), key=lambda item: item[1], reverse=True)
            }
        vocab_rev = {k: v for v, k in tokeniser.vocab.items()}
        wanted = [vocab_rev[i] for i in list(counts.keys())[:model_features]]
        # unwanted = [vocab_rev[i] for i in list(counts.keys())[model_features:]]
        model_state = json.loads(
            tokeniser.backend_tokenizer.model.__getstate__()
            )
        ref = pd.DataFrame(model_state["vocab"])
        remove = ref[~ref[0].isin(wanted)].index.tolist()

        # remove = remove[:2] + [2] + remove[2:]
        # for i in tqdm(sorted(remove, reverse=True)):
            # del model_state["vocab"][i]
        # model_class = getattr(models, model_state.pop("type"))
        # tokenizer.backend_tokenizer.model = model_class(**model_state)

        with open(tokeniser_path, 'r') as infile:
            tokeniser_file = json.load(infile)
        for i in tqdm(sorted(remove, reverse=True)):
            del tokeniser_file["model"]["vocab"][i]

        tokeniser_path_reduced = ".".join([
            "".join(tokeniser_path.split(".json")), str(model_features), "json"
            ])
        with open(tokeniser_path_reduced, 'w', encoding='utf-8') as f:
            json.dump(
                tokeniser_file, f, ensure_ascii=False, indent=4
                )

        special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
        print("USING MODIFIED TOKENISER:", tokeniser_path_reduced)
        tokeniser = PreTrainedTokenizerFast(
            tokenizer_file=tokeniser_path_reduced,
            special_tokens=special_tokens,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
            )
        data_collator = DataCollatorWithPadding(tokenizer=tokeniser,
                                                padding="longest")

    for i in dataset:
        if "input_ids" in dataset[i].features:
            dataset[i].features["input_ids"] = Value('int32')
        if "attention_mask" in dataset[i].features:
            dataset[i].features["attention_mask"] = Value('int8')
        if "token_type_ids" in dataset[i].features:
            dataset[i] = dataset[i].remove_columns("token_type_ids")
        if "input_str" in dataset[i].features:
            dataset[i] = dataset[i].remove_columns("input_str")
        # by default this will be "labels"
        if type(dataset[i].features[args.label_names[0]]) != ClassLabel:
            dataset[i] = dataset[i].class_encode_column(args.label_names[0])    

    # dataset["train"].features[args.label_names].names = ["NEG", "POS"]
    print("\nSAMPLE DATASET ENTRY:\n", dataset["train"][0], "\n")
    dataset = dataset.map(
        lambda data: tokeniser(data['feature'], padding=True), batched=True
        )

    col_torch = ['input_ids', 'attention_mask', args.label_names[0]]
    # col_torch = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    print(dataset)
    dataset.set_format(type='torch', columns=col_torch)
    # dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=1)
    # print("\nSAMPLE PYTORCH FORMATTED ENTRY:\n", next(iter(dataloader)))

    if model == "distilbert":
        config = DistilBertConfig(
            vocab_size=vocab_size,
            num_labels=2,
            # output_hidden_states=True,
            # output_attentions=True
            )
        model = DistilBertForSequenceClassification(config)
        model.resize_token_embeddings(len(tokeniser))
        model = model.to(device)
        def _model_init():
            model = DistilBertForSequenceClassification(config)
            model.resize_token_embeddings(len(tokeniser))
            model = model.to(device)
            return model
    if model == "longformer":
        config = LongformerConfig(
            vocab_size=vocab_size,
            num_labels=2,
            # output_hidden_states=True,
            # output_attentions=True
            )
        model = LongformerForSequenceClassification(config)
        model.resize_token_embeddings(len(tokeniser))
        model = model.to(device)
        def _model_init():
            model = LongformerForSequenceClassification(config)
            model.resize_token_embeddings(len(tokeniser))
            model = model.to(device)
            return model

    model_size = sum(t.numel() for t in model.parameters())
    print(f"\nDistilBert size: {model_size/1000**2:.1f}M parameters")
    tokeniser.pad_token = tokeniser.eos_token

    # hyperparameter tuning on 10% of original dataset following:
    # https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb

    # if hyperparameter sweep is provided or set to True,
    if os.path.exists(hyperparameter_sweep):
        print(" ".join(["Loading sweep hyperparameters from",
                       hyperparameter_sweep]))
        with open(hyperparameter_sweep, 'r') as infile:
            sweep_config = json.load(infile)
    else:
        warn("No sweep hyperparameters provided, do random search")
        # method
        sweep_config = {
            'name': 'random',
            'method': 'random',
            "metric": {
                "name": "eval/f1",
                "goal": "maximize"
                },
            'parameters': {
                'epochs': {
                    'values': [1, 2, 3, 4, 5]
                    },
                'dropout': {
                  "values": [0.15, 0.2, 0.25, 0.3, 0.4]
                },
                'batch_size': {
                    'values': [8, 16, 32, 64]
                    },
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-1
                },
                'weight_decay': {
                    'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                },
                'decay': {
                    'values': [1e-5, 1e-6, 1e-7]
                },
                'momentum': {
                    'values': [0.8, 0.9, 0.95]
                }
            },
            "early_terminate": {
                "type": "hyperband",
                "s": 2,
                "eta": 3,
                "max_iter": 27
            }
        }

    def _compute_metrics(eval_preds):
        """Compute metrics during the training run using transformers and wandb API.

        This is configured to capture metrics using the transformers `dataset` API,
        and upload the metrics to `wandb` for interactive logging and visualisation.
        Not intended for direct use, this is called by `transformers.Trainer()`.

        More information regarding evaluation metrics.
        - https://huggingface.co/course/chapter3/3?fw=pt
        - https://discuss.huggingface.co/t/log-multiple-metrics-while-training/8115/4
        - https://wandb.ai/matt24/vit-snacks-sweeps/reports/Hyperparameter-Search-with-W-B-Sweeps-for-Hugging-Face-Transformer-Models--VmlldzoyMTUxNTg0

        Args:
            eval_preds (torch): a tensor passed in as part of the training process.

        Returns:
            dict:

            A dictionary of metrics from the transformers `dataset` API.
            This is specifically configured for plotting `wandb` interactive plots.
        """
        metrics = dict()
        accuracy_metric = load_metric('accuracy')
        precision_metric = load_metric('precision')
        recall_metric = load_metric('recall')
        f1_metric = load_metric('f1')
        logits = eval_preds.predictions
        labels = eval_preds.label_ids
        preds = np.argmax(logits, axis=-1)
        y_probas = np.concatenate(
            (1 - preds.reshape(-1,1), preds.reshape(-1,1)), axis=1
            )
        class_names = dataset["train"].features[args.label_names[0]].names
        wandb.log({"roc_curve" : wandb.plot.roc_curve(
            labels, y_probas, labels=class_names
            )})
        wandb.log({"pr" : wandb.plot.pr_curve(
            labels, y_probas, labels=class_names, #classes_to_plot=None
            )})
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(
            probs=y_probas, y_true=labels, class_names=class_names
            )})
        metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
        metrics.update(precision_metric.compute(predictions=preds, references=labels, average='weighted'))
        metrics.update(recall_metric.compute(predictions=preds, references=labels, average='weighted'))
        metrics.update(f1_metric.compute(predictions=preds, references=labels, average='weighted'))
        return metrics

    if resume_sweep == None:
        sweep_id = wandb.sweep(sweep_config,
                               entity=entity_name,
                               project=project_name)
    else:
        sweep_id = resume_sweep

    # function needs to be defined internally to access namespace
    def _sweep_wandb(config=None):
        with wandb.init(
            group=group_name,
            job_type=group_name,
            settings=wandb.Settings(console='off', start_method='fork'),
            entity=entity_name,
            project=project_name,
            ):
            # set sweep configuration
            config = wandb.config
            api = wandb.Api(timeout=10000)
            sweep_dir = "/".join([args.output_dir, sweep_config["name"]])
            print("\nLOGGING OUTPUT TO:\n", sweep_dir, "\n")
            # set training arguments
            training_args = TrainingArguments(
                output_dir=sweep_dir,
        	    report_to='wandb',  # Turn on Weights & Biases logging
                num_train_epochs=config.epochs,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                save_strategy='epoch',
                evaluation_strategy='epoch',
                logging_strategy='epoch',
                load_best_model_at_end=True,
                remove_unused_columns=False,
                fp16=fp16,
                bf16=False,
            )
            # define training loop
            if partition_percent == 100:
                trainer = Trainer(
                    model_init=_model_init,
                    args=training_args,
                    tokenizer=tokeniser,
                    train_dataset=dataset['train'],
                    eval_dataset=dataset['valid'],
                    compute_metrics=_compute_metrics,
                    data_collator=data_collator,
                )                
            else:
                trainer = Trainer(
                    model_init=_model_init,
                    args=training_args,
                    tokenizer=tokeniser,
                    train_dataset=dataset['train'].shard(index=0, num_shards=int(100/partition_percent)),
                    eval_dataset=dataset['valid'].shard(index=0, num_shards=int(100/partition_percent)),
                    compute_metrics=_compute_metrics,
                    data_collator=data_collator,
                )
            # start training loop
            trainer.train()
            print("Saving model to:", wandb.run.dir)
            trainer.save_model(wandb.run.dir)
            with open(
                os.path.join(wandb.run.dir, "training_args.json"),
                "w",
                encoding="utf-8"
                ) as args_json:
                json.dump(
                    training_args.to_json_string(),
                    args_json,
                    ensure_ascii=False,
                    indent=4
                    )
            wandb.save(os.path.join(wandb.run.dir, "pytorch_model.bin"))
            wandb.save(os.path.join(wandb.run.dir, "training_args.json"))

    # allow passing of argument with variable outside namespace
    # wandb_train_func = functools.partial(_sweep_wandb, sweep_config)
    wandb.agent(sweep_id,
                function=_sweep_wandb,
                count=sweep_count,
                entity=entity_name,
                project=project_name)
    wandb.finish()

    # connect to the finished sweep using API
    api = wandb.Api()
    entity_project_id = "/".join([entity_name, project_name, sweep_id])
    sweep = api.sweep(entity_project_id)
    print("Entity / Project / Sweep ID:", sweep)

    # download metrics from all runs
    print("Get metrics from all runs")
    if group_name == None and any([entity_name, project_name]):
        runs = api.runs(path="/".join([entity_name, project_name]),)
    else:
        runs = api.runs(path="/".join([entity_name, project_name]),
                        filters={"group": group_name})

    get_run_metrics(runs, args.output_dir)

    # download best model file from the sweep
    runs = sorted(
        sweep.runs,
        key=lambda run: run.summary.get(metric_opt, 0),
        reverse=True
        )
    print("Get best model file from the sweep:", runs[0])
    score = runs[0].summary.get(metric_opt, 0)
    print(f"Best run {runs[0].name} with {metric_opt}={score}%")
    best_model = "/".join([args.output_dir, "model_files"])
    for i in runs[0].files():
        i.download(root=best_model, replace=True)
    print("\nBEST MODEL AND CONFIG FILES SAVED TO:\n", best_model)
    print("\nHYPERPARAMETER SWEEP END")

if __name__ == "__main__":
    _cite_me()
    main()
