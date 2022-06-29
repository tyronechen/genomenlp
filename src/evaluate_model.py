import argparse
import gzip
import os
from random import shuffle
from warnings import warn
import datasets
from datasets import ClassLabel, Dataset, DatasetDict, Value, load_dataset, load_metric
import pandas as pd
import torch
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, HfArgumentParser,\
    PreTrainedTokenizerFast, Trainer, TrainingArguments
import numpy as np

def load_data(infile_path: str):
    """Take a 🤗 dataset object, path as output and write files to disk"""
    if infile_path.endswith(".csv") or infile_path.endswith(".csv.gz"):
        return load_dataset("csv", data_files=infile_path)
    elif infile_path.endswith(".json"):
        return load_dataset("json", data_files=infile_path)
    elif infile_path.endswith(".parquet"):
        return load_dataset("parquet", data_files=infile_path)

def main():
    parser = argparse.ArgumentParser(
        description='Take HuggingFace🤗 dataset and evaluate. \
        See 🤗 documentation here for reference: \
        https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb'
        )
    parser.add_argument('test', type=str,
                        help='path to [ csv | csv.gz | json | parquet ] file')
    parser.add_argument('format', type=str,
                        help='specify input file type [ csv | json | parquet ]')
    parser.add_argument('model_path', type=str,
                        help='path to model directory to load data from')
    parser.add_argument('-m', '--metric', type=str, default="f1",
                        help='choose evaluation metric [ f1 ]')

    args = parser.parse_args()
    test = args.test
    format = args.format
    model_path = args.model_path
    metric = load_metric(args.metric)

    infile_paths = dict()
    infile_paths["test"] = test
    dataset = load_dataset(format, data_files=infile_paths)["test"]

    args_train = torch.load("/".join([model_path, "training_args.bin"]))
    args_train.metric_for_best_model = metric

    special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    tokeniser = PreTrainedTokenizerFast(
        tokenizer_file="/".join([model_path, "tokenizer.json"]),
        special_tokens=special_tokens,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        )
    tokeniser.pad_token = tokeniser.eos_token

    def _compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=DistilBertForSequenceClassification.from_pretrained(model_path),
        args=args_train,
        # train_dataset=dataset["train"],
        eval_dataset=dataset,
        tokenizer=tokeniser,
        compute_metrics=_compute_metrics
    )
    eval = trainer.evaluate()
    print(eval)

    # TODO: hyperparameter search

if __name__ == "__main__":
    main()
