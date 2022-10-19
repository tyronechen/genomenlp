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
from utils import _compute_metrics, load_args_json, load_args_cmd
import wandb

def main():
    parser = HfArgumentParser(
        [TrainingArguments], description='Take HuggingFace dataset and train.\
          Arguments match that of TrainingArguments, with the addition of \
         [ train, test, valid, tokeniser_path, vocab_size, hyperparameter_cpus \
         hyperparameter_tune hyperparameter_file no_shuffle, wandb_off ]. See: \
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
                        NOTE: if given, this overrides all HfTrainingArguments!')
    parser.add_argument('-e', '--entity_name', type=str, default="",
                        help='provide wandb team name (if available). \
                        NOTE: has no effect if wandb is disabled.')
    parser.add_argument('-p', '--project_name', type=str, default="",
                        help='provide wandb project name (if available). \
                        NOTE: has no effect if wandb is disabled.')
    parser.add_argument('-g', '--group_name', type=str, default="train",
                        help='provide wandb group name (if desired).')
    parser.add_argument('-o', '--metric_opt', type=str, default="eval/f1",
                        help='score to maximise [ eval/accuracy | \
                        eval/validation | eval/loss | eval/precision | \
                        eval/recall ] (DEFAULT: eval/f1)')
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
    assert type(args_train) == transformers.training_args.TrainingArguments, \
        "Must be instance of transformers.training_args.TrainingArguments"

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
        # disable_tqdm=args.disable_tqdm,
    )

    print(trainer)
    train = trainer.train()
    print(train)
    model_out = "/".join([args.output_dir, "model_files"])
    print("Saving model to:", model_out)
    trainer.save_model(model_out)
    # eval = trainer.evaluate(_compute_metrics)
    # print(eval)
    eval_results = trainer.evaluate()
    print(eval_results)
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

    print("Model file:", runs[0])
    score = runs[0].summary.get(metric_opt, 0)
    print(f"Run {runs[0].name} with {metric_opt}={score}%")
    best_model = "/".join([args.output_dir, "model_files"])
    for i in runs[0].files():
        i.download(root=best_model, replace=True)
    print("\nMODEL AND CONFIG FILES SAVED TO:\n", best_model)
    print("\nTRAIN SWEEP END")

if __name__ == "__main__":
    main()
