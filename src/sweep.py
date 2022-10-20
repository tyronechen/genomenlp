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
from utils import _compute_metrics, load_args_json
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
    parser.add_argument('-o', '--output_dir', type=str, default="./sweep_out",
                        help='specify path for output (DEFAULT: ./sweep_out)')
    parser.add_argument('-d', '--device', type=str, default="auto",
                        help='choose device [ cpu | cuda:0 ] (DEFAULT: detect)')
    parser.add_argument('-s', '--vocab_size', type=int, default=32000,
                        help='vocabulary size for model configuration')
    parser.add_argument('-w', '--hyperparameter_sweep', type=str, default="",
                        help='run a hyperparameter sweep with config from file')
    parser.add_argument('-n', '--sweep_count', type=int, default=8,
                        help='run n hyperparameter sweeps (DEFAULT: 64)')
    parser.add_argument('-e', '--entity_name', type=str, default="",
                        help='provide wandb team name (if available).')
    parser.add_argument('-p', '--project_name', type=str, default="",
                        help='provide wandb project name (if available).')
    parser.add_argument('-g', '--group_name', type=str, default="sweep",
                        help='provide wandb group name (if desired).')
    parser.add_argument('-c', '--metric_opt', type=str, default="eval/f1",
                        help='score to maximise [ eval/accuracy | \
                        eval/validation | eval/loss | eval/precision | \
                        eval/recall ] (DEFAULT: eval/f1)')
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
    output_dir = args.output_dir
    fp16 = args.fp16_off
    if wandb_state is True:
        wandb.login()
        args.report_to = "wandb"
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

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
    if "token_type_ids" in dataset:
        dataset = dataset.remove_columns("token_type_ids")
    dataset = dataset.class_encode_column(args.label_names[0])
    # dataset["train"].features[args.label_names].names = ["NEG", "POS"]
    print("\nSAMPLE DATASET ENTRY:\n", dataset["train"][0], "\n")

    col_torch = ['input_ids', 'attention_mask', args.label_names[0]]
    # col_torch = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    print(dataset)
    dataset.set_format(type='torch', columns=col_torch)
    dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=1)
    print("\nSAMPLE PYTORCH FORMATTED ENTRY:\n", next(iter(dataloader)))

    if model == "distilbert":
        config = DistilBertConfig(vocab_size=vocab_size, num_labels=2)
        model = DistilBertForSequenceClassification(config).to(device)
        def _model_init():
            return DistilBertForSequenceClassification(config).to(device)
    if model == "longformer":
        config = LongformerConfig(vocab_size=vocab_size, num_labels=2)
        model = LongformerForSequenceClassification(config).to(device)
        def _model_init():
            return LongformerForSequenceClassification(config).to(device)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"\nDistilBert size: {model_size/1000**2:.1f}M parameters")
    tokeniser.pad_token = tokeniser.eos_token

    # hyperparameter tuning on 10% of original dataset following:
    # https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb

    # if hyperparameter sweep is provided or set to True,
    if wandb_state == True:
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

        sweep_id = wandb.sweep(sweep_config,
                               entity=entity_name,
                               project=project_name)

        # function needs to be defined internally to access namespace
        def _sweep_wandb(config=None):
            with wandb.init(
                group=group_name,
                job_type=group_name,
                config=config,
                settings=wandb.Settings(console='off', start_method='fork'),
                entity=entity_name,
                project=project_name
                ):
                # set sweep configuration
                config = wandb.config
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
                trainer = Trainer(
                    model_init=_model_init,
                    args=training_args,
                    tokenizer=tokeniser,
                    train_dataset=dataset['train'],
                    eval_dataset=dataset['valid'],#.shard(index=1, num_shards=10),
                    compute_metrics=_compute_metrics,
                    data_collator=data_collator,
                )
                # start training loop
                trainer.train()
                print("Saving model to:", wandb.run.dir)
                trainer.save_model(wandb.run.dir)
                wandb.save(os.path.join(wandb.run.dir, "pytorch_model.bin"))

        # allow passing of argument with variable outside namespace
        # wandb_train_func = functools.partial(_sweep_wandb, sweep_config)
        wandb.agent(sweep_id,
                    # function=wandb_train_func,
                    function=_sweep_wandb,
                    count=sweep_count)
        wandb.finish()

        # connect to the finished sweep using API
        api = wandb.Api()
        entity_project_id = "/".join([entity_name, project_name, sweep_id])
        sweep = api.sweep(entity_project_id)
        print("Entity / Project / Sweep ID:", sweep)

        # download metrics from all runs
        print("Get metrics from all runs")
        runs = api.runs("/".join([entity_name, project_name]))
        summary_list, config_list, name_list = [], [], []
        for run in runs:
            # .summary contains the output keys/values for metrics
            #  We call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            config_list.append(
                {k: v for k,v in run.config.items()
                 if not k.startswith('_')})
            # .name is the human-readable name of the run
            name_list.append(run.name)
        runs_df = pd.DataFrame({
            "summary": summary_list,
            "config": config_list,
            "name": name_list
            })
        runs_df.to_csv("/".join([args.output_dir, "metrics.csv"]))

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
    main()
