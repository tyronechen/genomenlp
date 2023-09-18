#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# cross-validate
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
from utils import load_args_json, load_args_cmd, get_run_metrics, _cite_me
# import nevergrad as ng
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import download_data, \
    build_compute_metrics_fn
import wandb

def main():
    # parser = HfArgumentParser(
    #     [TrainingArguments], description='Take HuggingFace dataset and train. \
    #      Arguments match that of TrainingArguments, with the addition of \
    #      [ train, test, valid, tokeniser_path, vocab_size, hyperparameter_file,\
    #       model, device, kfolds, entity_name, group_name, project_name, \
    #      config_from_run, metric_opt, override_output_dir, no_shuffle, \
    #      wandb_off ]. See: \
    #      https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/trainer#transformers.TrainingArguments'
    #     )
    parser = argparse.ArgumentParser(
        description='Take HuggingFace dataset and perform cross validation.'
    )
    parser.add_argument('train', type=str,
                        help='path to [ csv | csv.gz | json | parquet ] file')
    parser.add_argument('format', type=str,
                        help='specify input file type [ csv | json | parquet ]')
    parser.add_argument('--tokeniser_path', type=str, default="",
                        help='path to tokeniser.json file to load data from')
    parser.add_argument('-t', '--test', type=str, default=None,
                        help='path to [ csv | csv.gz | json | parquet ] file')
    parser.add_argument('-v', '--valid', type=str, default=None,
                        help='path to [ csv | csv.gz | json | parquet ] file')
    parser.add_argument('-m', '--model_path', type=str, default=None,
                        help='path to pretrained model dir. this should contain\
                         files such as [ pytorch_model.bin, config.yaml, \
                         tokeniser.json, etc ]')
    parser.add_argument('-o', '--output_dir', type=str, default="./cval_out",
                        help='specify path for output (DEFAULT: ./cval_out)')    
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='choose device [ cpu | cuda:0 ] (DEFAULT: detect)')
    parser.add_argument('-s', '--vocab_size', type=int, default=32000,
                        help='vocabulary size for model configuration')
    parser.add_argument('-f', '--hyperparameter_file', type=str, default="",
                        help='provide torch.bin or json file of hyperparameters. \
                        NOTE: if given, this overrides all HfTrainingArguments! \
                        This is overridden by --config_from_run!')
    parser.add_argument('-l', '--label_names', type=str, default="", nargs="+",
                        help='provide column with label names (DEFAULT: "").')    
    parser.add_argument('-k', '--kfolds', type=int, default=8,
                        help='run n number of kfolds (DEFAULT: 8)')
    parser.add_argument('-e', '--entity_name', type=str, default="",
                        help='provide wandb team name (if available).')
    parser.add_argument('-g', '--group_name', type=str, default="crossval",
                        help='provide wandb group name (if desired).')
    parser.add_argument('-p', '--project_name', type=str, default="",
                        help='provide wandb project name (if available).')
    parser.add_argument('-c', '--config_from_run', type=str, default=None,
                        help='load arguments from existing wandb run. \
                        NOTE: if given, this overrides --hyperparameter_file!')
    parser.add_argument('--metric_opt', type=str, default="eval/f1",
                        help='score to maximise [ eval/accuracy | \
                        eval/validation | eval/loss | eval/precision | \
                        eval/recall ] (DEFAULT: eval/f1)')
    parser.add_argument('--overwrite_output_dir', action="store_true",
                        help='overwrite output directory (DEFAULT: OFF)')
    parser.add_argument('--no_shuffle', action="store_false",
                        help='turn off random shuffling (DEFAULT: SHUFFLE)')
    parser.add_argument('--wandb_off', action="store_false",
                        help='run hyperparameter tuning using the wandb api \
                        and log training in real time online (DEFAULT: ON)')

    args = parser.parse_args()
    train = args.train
    format = args.format
    model_path = args.model_path
    device = args.device
    test = args.test
    valid = args.valid
    tokeniser_path = args.tokeniser_path
    hyperparameter_file = args.hyperparameter_file
    kfolds = args.kfolds
    vocab_size = args.vocab_size
    shuffle = args.no_shuffle
    wandb_state = args.wandb_off
    entity_name = args.entity_name
    group_name = args.group_name
    project_name = args.project_name
    metric_opt = args.metric_opt
    main_output_dir = args.output_dir
    overwrite_output_dir = args.overwrite_output_dir
    config_from_run = args.config_from_run
    if wandb_state is True:
        wandb.login()
        args.report_to = "wandb"
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
        tokeniser.pad_token = tokeniser.eos_token

    if config_from_run != None:
        run_id = config_from_run
        api = wandb.Api(timeout=10000)
        run = api.run(run_id)
        [i.download(root=main_output_dir, replace=True) for i in run.files()]
        hyperparameter_file = "/".join([main_output_dir, "training_args.bin"])
        warn("".join([
            "Loading existing hyperparameters from: ", run_id, "!",
            "This overrides all HfTrainingArguments, including",
            " --hyperparameter_file and --tokeniser_path!"
            ]))
        model_path = main_output_dir

        tokeniser_path = "/".join([main_output_dir, "tokenizer.json"])
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
        tokeniser.pad_token = tokeniser.eos_token

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

    # dataset["train"].features[args.label_names].names = ["NEG", "POS"]
    print("\nSAMPLE DATASET ENTRY:\n", dataset["train"][0], "\n")
    dataset = dataset.map(
        lambda data: tokeniser(data['feature'], padding=True), batched=True
        )
    col_torch = ['input_ids', 'attention_mask', args.label_names[0]]
    # col_torch = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    print(dataset)
    dataset.set_format(type='torch', columns=col_torch)
    dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=1)
    print("\nSAMPLE PYTORCH FORMATTED ENTRY:\n", next(iter(dataloader)))

    # regarding evaluation metrics:
    # https://huggingface.co/course/chapter3/3?fw=pt
    # https://discuss.huggingface.co/t/log-multiple-metrics-while-training/8115/4
    # https://wandb.ai/matt24/vit-snacks-sweeps/reports/Hyperparameter-Search-with-W-B-Sweeps-for-Hugging-Face-Transformer-Models--VmlldzoyMTUxNTg0

    if os.path.exists(model_path):
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        def _model_init():
            return AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        raise OSError("Cross validation requires a pretrained model!")

    model_size = sum(t.numel() for t in model.parameters())
    print(f"\nDistilBert size: {model_size/1000**2:.1f}M parameters")

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
    if overwrite_output_dir == True:
        warn(" ".join(["\nOVERRIDE ARGS, OUTPUT TO:", main_output_dir, "\n"]))
        args_train.output_dir = main_output_dir
    else:
        warn(" ".join(["\nOUTPUT DIR NOT OVERRIDEN:", args_train.output_dir, "\n"]))
    assert type(args_train) == transformers.training_args.TrainingArguments, \
        "Must be instance of transformers.training_args.TrainingArguments"

    # select the number of kfolds
    folds = StratifiedKFold(n_splits=args.kfolds)
    # split the dataset based off of the labels.
    # use np.zeros() here since it only works off of indices, we want the labels
    splits = folds.split(
        np.zeros(dataset["train"].num_rows), dataset["train"][args.label_names[0]]
        )

    fold_count = 0
    for train_idxs, val_idxs in splits:
        fold_dataset = DatasetDict({
            "train": dataset["train"].select(train_idxs),
            "valid": dataset["train"].select(val_idxs),
            "test": dataset["valid"]
        })
        fold_count += 1
        wandb.init(
            group=group_name,
            job_type=group_name,
            settings=wandb.Settings(console='off', start_method='fork'),
            entity=entity_name,
            project=project_name,
            config=args_train,
            reinit=True,
            )
        wandb.config.update({"fold_count": fold_count})
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
            class_names = fold_dataset["train"].features[args.label_names[0]].names
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
        eval_results = trainer.evaluate()
        print(eval_results)
        wandb.save(os.path.join(wandb.run.dir, "pytorch_model.bin"))
    wandb.finish()

    api = wandb.Api()
    entity_project_id = "/".join([entity_name, project_name])

    print("Entity / Project / Group ID:", entity_project_id, args.group_name)

    # download metrics from all runs
    print("Get metrics from all folds")
    if group_name == None and any([entity_name, project_name]):
        runs = api.runs(path="/".join([entity_name, project_name]),)
    else:
        runs = api.runs(path="/".join([entity_name, project_name]),
                        filters={"group": group_name})

    scores = get_run_metrics(runs, "cval")
    # scores = pd.DataFrame(scores.summary.apply(lambda x: x[metric_opt]))
    # scores.columns = ["roc_auc_scores"]
    # scores.to_csv("."join(["roc_auc_scores"]))

    # identify best model file from the sweep
    runs = sorted(
        runs, key=lambda run: run.summary.get(metric_opt, 0), reverse=True
        )
    # NOTE: it doesnt make sense to download the best model in cross validation
    score = runs[0].summary.get(metric_opt, 0)
    print(f"Best fold {runs[0].name} with {metric_opt}={score}%")
    print("\nCROSS VALIDATION END")

if __name__ == "__main__":
    main()
    _cite_me()
