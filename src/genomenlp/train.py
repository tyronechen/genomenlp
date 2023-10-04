#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# conduct train
import argparse
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
from tokenizers import SentencePieceUnigramTokenizer
from tqdm import tqdm
import transformers
from transformers import AutoModelForSequenceClassification, \
    DataCollatorWithPadding, \
    DistilBertConfig, DistilBertForSequenceClassification, HfArgumentParser, \
    LongformerConfig, LongformerForSequenceClassification, \
    PreTrainedTokenizerFast, Trainer, TrainingArguments, set_seed
from transformers.training_args import ParallelMode
from utils import get_run_metrics, load_args_json, load_args_cmd, _cite_me
import wandb

def main():
    parser = HfArgumentParser(
        [TrainingArguments], description='Take HuggingFace dataset and train.\
          Arguments match that of TrainingArguments, with the addition of \
         [ train, test, valid, tokeniser_path, vocab_size, model, device, \
          entity_name, project_name, group_name, config_from_run, metric_opt, \
          hyperparameter_file, no_shuffle, wandb_off, override_output_dir ]. See: \
         https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/trainer#transformers.TrainingArguments'
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
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='choose device [ cpu | cuda:0 ] (DEFAULT: detect)')
    parser.add_argument('-s', '--vocab_size', type=int, default=32000,
                        help='vocabulary size for model configuration')
    parser.add_argument('-f', '--hyperparameter_file', type=str, default="",
                        help='provide torch.bin or json file of hyperparameters. \
                        NOTE: if given, this overrides all HfTrainingArguments! \
                        This is overridden by --config_from_run!')
    parser.add_argument('-e', '--entity_name', type=str, default="",
                        help='provide wandb team name (if available). \
                        NOTE: has no effect if wandb is disabled.')
    parser.add_argument('-p', '--project_name', type=str, default="",
                        help='provide wandb project name (if available). \
                        NOTE: has no effect if wandb is disabled.')
    parser.add_argument('-g', '--group_name', type=str, default="train",
                        help='provide wandb group name (if desired).')
    parser.add_argument('-c', '--config_from_run', type=str, default=None,
                        help='load arguments from existing wandb run. \
                        NOTE: if given, this overrides --hyperparameter_file!')
    parser.add_argument('--metric_opt', type=str, default="eval/f1",
                        help='score to maximise [ eval/accuracy | \
                        eval/validation | eval/loss | eval/precision | \
                        eval/recall ] (DEFAULT: eval/f1)')
    parser.add_argument('--override_output_dir', action="store_true",
                        help='override output directory (DEFAULT: OFF)')
    parser.add_argument('--no_shuffle', action="store_false",
                        help='turn off random shuffling (DEFAULT: SHUFFLE)')
    parser.add_argument('--wandb_off', action="store_false",
                        help='log training in real time online (DEFAULT: ON)')

    args = parser.parse_args()
    train = args.train
    format = args.format
    model = args.model
    device = args.device
    test = args.test
    valid = args.valid
    tokeniser_path = args.tokeniser_path
    hyperparameter_file = args.hyperparameter_file
    vocab_size = args.vocab_size
    shuffle = args.no_shuffle
    wandb_state = args.wandb_off
    entity_name = args.entity_name
    project_name = args.project_name
    group_name = args.group_name
    metric_opt = args.metric_opt
    main_output_dir = args.output_dir
    override_output_dir = args.override_output_dir
    config_from_run = args.config_from_run
    if wandb_state is True:
        wandb.login()
        args.report_to = "wandb"
    else:
        args.report_to = None
    if device == None:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

    if device == "cpu":
        fp16 = False
    else:
        fp16 = True
    api = wandb.Api(timeout=10000)
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
        print(labels)
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

    if config_from_run != None:
        # run_id = "/".join([entity_name, project_name, config_from_run])
        run_id = config_from_run
        api = wandb.Api()
        run = api.run(run_id)
        run.file("training_args.bin").download(root=config_from_run, replace=True)
        hyperparameter_file = "/".join([config_from_run, "training_args.bin"])
        warn("".join([
            "Loading existing hyperparameters from: ", run_id, "!",
            "This overrides all HfTrainingArguments AND --hyperparameter_file!"
            ]))

    if os.path.exists(hyperparameter_file):
        warn("".join([
            "Loading existing hyperparameters from: ", hyperparameter_file,
            "This overrides all HfTrainingArguments!"
            ]))
        if hyperparameter_file.endswith(".json"):
            args_train = load_args_json(hyperparameter_file)
        if hyperparameter_file.endswith(".bin"):
            args_train = torch.load(hyperparameter_file)
    else:
        args_train = load_args_cmd(args)
    if override_output_dir == True:
        warn(" ".join(["\nOVERRIDE ARGS, OUTPUT TO:", main_output_dir, "\n"]))
        args_train.output_dir = main_output_dir
    else:
        warn(" ".join(["\nOUTPUT DIR NOT OVERRIDEN:", args_train.output_dir, "\n"]))
    assert type(args_train) == transformers.training_args.TrainingArguments, \
        "Must be instance of transformers.training_args.TrainingArguments"

    print("\n\n", args_train, "\n\n")

    wandb.init(
        group=group_name,
        job_type=group_name,
        settings=wandb.Settings(console='off', start_method='fork'),
        entity=entity_name,
        project=project_name,
        config=args_train,
        )

    # hyperparameter tuning on 10% of original datasett following:
    # https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb
    trainer = Trainer(
        model_init=_model_init,
        tokenizer=tokeniser,
        args=args_train,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=_compute_metrics,
        data_collator=data_collator,
    )

    print(trainer)
    train = trainer.train()
    print(train)
    model_out = "/".join([args_train.output_dir, "model_files"])
    print("\nSAVING MODEL TO:", model_out, "\n")
    trainer.save_model(model_out)
    eval_results = trainer.evaluate()
    print(eval_results)
    wandb.save(os.path.join(wandb.run.dir, "pytorch_model.bin"))
    wandb.finish()

    api = wandb.Api()
    entity_project_id = "/".join([entity_name, project_name])

    print("\nEntity / Project / Group ID:", entity_project_id, args.group_name)

    # download metrics from all runs
    print("\nGet metrics from all runs\n")
    if group_name == None and any([entity_name, project_name]):
        runs = api.runs(path="/".join([entity_name, project_name]),)
    else:
        runs = api.runs(path="/".join([entity_name, project_name]),
                        filters={"group": group_name})

    get_run_metrics(runs, args.output_dir)

    print("\nModel file:", runs[0], "\n")
    score = runs[0].summary.get(metric_opt, 0)
    print(f"Run {runs[0].name} with {metric_opt}={score}%")
    best_model = "/".join([args_train.output_dir, "model_files"])
    for i in runs[0].files():
        i.download(root=best_model, replace=True)
    print("\nMODEL AND CONFIG FILES SAVED TO:\n", best_model)
    print("\nTRAIN SWEEP END")

if __name__ == "__main__":
    main()
    _cite_me()
