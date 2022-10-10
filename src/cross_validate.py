import argparse
import collections
import functools
import gzip
import multiprocessing
import os
import random
import sys
from random import shuffle
from warnings import warn
from datasets import ClassLabel, Dataset, DatasetDict, Value, load_dataset, load_metric
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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
from utils import _compute_metrics
# import nevergrad as ng
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import download_data, \
    build_compute_metrics_fn
import wandb

def main():
    parser = HfArgumentParser(
        [TrainingArguments], description='Take HuggingFace dataset and train.\
          Arguments match that of TrainingArguments, with the addition of \
         [ train, test, valid, tokeniser_path, vocab_size, hyperparameter_cpus,\
           no_shuffle, wandb_off ]. See: \
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
    parser.add_argument('-k', '--kfolds', type=int, default=8,
                        help='run n number of kfolds (DEFAULT: 8)')
    parser.add_argument('-e', '--entity_name', type=str, default="",
                        help='provide wandb team name (if available).')
    parser.add_argument('-g', '--group_name', type=str, default="crossval",
                        help='provide wandb group name (if desired).')
    parser.add_argument('-p', '--project_name', type=str, default="",
                        help='provide wandb project name (if available).')
    parser.add_argument('--no_shuffle', action="store_false",
                        help='turn off random shuffling (DEFAULT: SHUFFLE)')
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
    kfolds = args.kfolds
    vocab_size = args.vocab_size
    shuffle = args.no_shuffle
    wandb_state = args.wandb_off
    entity_name = args.entity_name
    group_name = args.group_name
    project_name = args.project_name
    if wandb_state is True:
        wandb.login()
    if device == None:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

    if device == "cpu":
        fp16 = False
    else:
        fp16 = True

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

    # regarding evaluation metrics:
    # https://huggingface.co/course/chapter3/3?fw=pt
    # https://discuss.huggingface.co/t/log-multiple-metrics-while-training/8115/4
    # https://wandb.ai/matt24/vit-snacks-sweeps/reports/Hyperparameter-Search-with-W-B-Sweeps-for-Hugging-Face-Transformer-Models--VmlldzoyMTUxNTg0

    args_train = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        per_gpu_train_batch_size=args.per_gpu_train_batch_size,
        per_gpu_eval_batch_size=args.per_gpu_eval_batch_size,
        evaluation_strategy=args.evaluation_strategy, #"steps",
        eval_steps=args.eval_steps, #5_000,
        logging_steps=args.logging_steps, #5_000,
        gradient_accumulation_steps=args.gradient_accumulation_steps, #8,
        num_train_epochs=args.num_train_epochs, #1,
        weight_decay=args.weight_decay, #0.1,
        warmup_steps=args.warmup_steps, #1_000,
        local_rank=args.local_rank,
        lr_scheduler_type=args.lr_scheduler_type, #"cosine",
        learning_rate=args.learning_rate, #5e-4,
        save_steps=args.save_steps, #5_000,
        skip_memory_metrics=False,
        fp16=fp16, #True,
        push_to_hub=args.push_to_hub, #False,
        label_names=args.label_names, #["labels"],
        report_to=args.report_to,
        run_name=args.run_name,
    )

    # select the number of kfolds
    folds = StratifiedKFold(n_splits=args.kfolds)
    # split the dataset based off of the labels.
    # use np.zeros() here since it only works off of indices, we want the labels
    splits = folds.split(
        np.zeros(dataset["train"].num_rows), dataset["train"]["label"]
        )

    for train_idxs, val_idxs in splits:
        fold_dataset = DatasetDict({
            "train": dataset["train"].select(train_idxs),
            "valid": dataset["train"].select(val_idxs),
            "test": dataset["valid"]
        })
        wandb.init(
            group=args.group_name,
            job_type=args.group_name,
            settings=wandb.Settings(console='off', start_method='fork'),
            entity=args.entity_name
            )
        # define training loop
        trainer = Trainer(
            model_init=_model_init,
            args=args_train,
            tokenizer=tokeniser,
            train_dataset=fold_dataset['train'],
            eval_dataset=fold_dataset['valid'],#.shard(index=1, num_shards=10),
            compute_metrics=_compute_metrics,
            data_collator=data_collator,
        )
        # start training loop
        trainer.train()
        print("Saving model to:", wandb.run.dir)
        trainer.save_model(wandb.run.dir)
        wandb.save(os.path.join(wandb.run.dir, "pytorch_model.bin"))
    wandb.finish()

    api = wandb.Api()
    entity_project_id = "/".join([entity_name, project_name])
    runs = api.runs(path="/".join([entity_name, project_name]),
                    filters={"group_name": args.group_name})

    print("Entity / Project / Group ID:", entity_project_id, args.group_name)

    # download metrics from all runs
    print("Get metrics from all runs")
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

    # TODO: to load training args as a json file of args later
    # if os.path.exists(hyperparameter_sweep):
    #     print(" ".join(["Loading sweep hyperparameters from",
    #                    hyperparameter_sweep]))
    #     with open(hyperparameter_sweep, 'r') as infile:
    #         sweep_config = json.load(infile)

    # download best model file from the sweep
    # these two lines dont work for some reason
    metric_opt = sweep_config["metric"]["name"]
    runs = sorted(
        runs, key=lambda run: run.summary.get(metric_opt, 0), reverse=True
        )
    # NOTE: it doesnt make sense to download the best model in cross validation
    # print("Get best model file from the sweep:", runs[0])
    score = runs[0].summary.get(metric_opt, 0)
    print(f"Best run {runs[0].name} with {metric_opt}={score}%")
    # best_model = "/".join([args.output_dir, "model_files"])
    # for i in runs[0].files():
    #     i.download(root=best_model, replace=True)
    # print("\nBEST MODEL AND CONFIG FILES SAVED TO:\n", best_model)
    # print("\nCROSS VALIDATION END")

if __name__ == "__main__":
    main()
