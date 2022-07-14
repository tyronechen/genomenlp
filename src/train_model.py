import argparse
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
from transformers import AutoModelForSequenceClassification, DistilBertConfig, \
    DistilBertForSequenceClassification, HfArgumentParser, \
    PreTrainedTokenizerFast, Trainer, TrainingArguments, set_seed
from transformers.training_args import ParallelMode
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import download_data, \
    build_compute_metrics_fn
from ray.tune.schedulers import PopulationBasedTraining

def load_data(infile_path: str):
    """Take a 🤗 dataset object, path as output and write files to disk"""
    if infile_path.endswith(".csv") or infile_path.endswith(".csv.gz"):
        return load_dataset("csv", data_files=infile_path)
    elif infile_path.endswith(".json"):
        return load_dataset("json", data_files=infile_path)
    elif infile_path.endswith(".parquet"):
        return load_dataset("parquet", data_files=infile_path)

def main():
    parser = HfArgumentParser(
        [TrainingArguments], description='Take HuggingFace dataset and train.\
          Arguments match that of TrainingArguments, with the addition of \
         [ train, test, valid, tokeniser_path, no_shuffle, wandb_off ]. See: \
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
    parser.add_argument('-c', '--hyperparameter_cpus', type=int, default=1,
                        help='number of cpus for hyperparameter tuning. \
                        NOTE: has no effect if --hyperparameter tune is False')
    parser.add_argument('-x', '--hyperparameter_tune', type=bool, default=False,
                        help='enable or disable hyperparameter tuning')
    parser.add_argument('-f', '--hyperparameter_file', type=str, default="",
                        help='provide a json file of hyperparameters. \
                        NOTE: if given, this overrides --hyperparameter_tune!')
    parser.add_argument('--no_shuffle', action="store_false",
                        help='turn off random shuffling (DEFAULT: SHUFFLE)')
    parser.add_argument('--wandb_off', action="store_false",
                        help='log training in real time online (DEFAULT: ON)')

    args = parser.parse_args()
    train = args.train
    format = args.format
    test = args.test
    valid = args.valid
    tokeniser_path = args.tokeniser_path
    hyperparameter_cpus = args.hyperparameter_cpus
    hyperparameter_tune = args.hyperparameter_tune
    hyperparameter_file = args.hyperparameter_file
    shuffle = args.no_shuffle
    wandb = args.wandb_off
    if wandb is False:
        os.environ["WANDB_DISABLED"] = "true"

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

    infile_paths = dict()
    infile_paths["train"] = train
    if test != None:
        infile_paths["test"] = test
    if valid != None:
        infile_paths["valid"] = valid
    dataset = load_dataset(format, data_files=infile_paths)

    print("\nSAMPLE DATASET ENTRY:\n", dataset["train"][0], "\n")

    col_torch = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    print(dataset)
    dataset.set_format(type='torch', columns=col_torch)
    dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=1)
    print("\nSAMPLE PYTORCH FORMATTED ENTRY:\n", next(iter(dataloader)))

    config = DistilBertConfig(vocab_size=32000, num_labels=2)
    model = DistilBertForSequenceClassification(config)

    def _model_init():
        return DistilBertForSequenceClassification(config)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"\nDistilBert size: {model_size/1000**2:.1f}M parameters")
    tokeniser.pad_token = tokeniser.eos_token
    args_train = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy=args.evaluation_strategy, #"steps",
        eval_steps=args.eval_steps, #5_000,
        logging_steps=args.logging_steps, #5_000,
        gradient_accumulation_steps=args.gradient_accumulation_steps, #8,
        num_train_epochs=args.num_train_epochs, #1,
        weight_decay=args.weight_decay, #0.1,
        warmup_steps=args.warmup_steps, #1_000,
        lr_scheduler_type=args.lr_scheduler_type, #"cosine",
        learning_rate=args.learning_rate, #5e-4,
        save_steps=args.save_steps, #5_000,
        fp16=args.fp16, #True,
        push_to_hub=args.push_to_hub, #False,
        label_names=args.label_names, #["labels"],
    )

    # regarding evaluation metrics:
    # https://huggingface.co/course/chapter3/3?fw=pt
    # https://discuss.huggingface.co/t/log-multiple-metrics-while-training/8115/4
    def _compute_metrics(eval_pred):
        accuracy = load_metric("accuracy")
        f1 = load_metric("f1")
        matthews_correlation = load_metric("matthews_correlation")
        precision = load_metric("precision")
        recall = load_metric("recall")
        roc_auc = load_metric("roc_auc")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy.compute(
            predictions=predictions, references=labels
            )["accuracy"]
        f1 = f1.compute(
            predictions=predictions, references=labels
            )["f1"]
        matthews_correlation = matthews_correlation.compute(
            predictions=predictions, references=labels
            )["matthews_correlation"]
        precision = precision.compute(
            predictions=predictions, references=labels
            )["precision"]
        recall = recall.compute(
            predictions=predictions, references=labels
            )["recall"]
        roc_auc = roc_auc.compute(
            predictions=predictions, references=labels
            )["roc_auc"]
        return {
            "accuracy": accuracy, "f1": f1,
            "matthews_correlation": matthews_correlation,
            "precision": precision, "recall": recall, "roc_auc": roc_auc,
            }

    f1 = load_metric("f1")
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return f1.compute(predictions=predictions, references=labels)

    # def _compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     predictions = np.argmax(predictions, axis=1)
    #     return metric.compute(predictions=predictions, references=labels)

    # hyperparameter tuning on 10% of original datasett following:
    # https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb
    trainer = Trainer(
        model_init=_model_init,
        tokenizer=tokeniser,
        args=args_train,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=_compute_metrics,
        # disable_tqdm=args.disable_tqdm,
    )

    if hyperparameter_file != "":
        if os.path.exists(hyperparameter_file):
            warn(" ".join(["Loading existing hyperparameters from",
                           hyperparameter_file]))
            with open(hyperparameter_file, 'r') as infile:
                hparams = json.load(infile)
            trainer.learning_rate = hparams["learning_rate"]
            trainer.num_train_epochs = hparams["num_train_epochs"]
            trainer.per_device_train_batch_size = hparams["per_device_train_batch_size"]
            hyperparameter_tune = False
            warn("\nHyperparameter file was provided, skipping tuning...\n")
        else:
            warn("\nNo hyperparameter file found! Using default settings...\n")

    if hyperparameter_tune is True:
        trainer = Trainer(
            model_init=_model_init,
            tokenizer=tokeniser,
            args=args_train,
            train_dataset=dataset["train"].shard(index=1, num_shards=100),
            eval_dataset=dataset["valid"],
            # disable_tqdm=args.disable_tqdm,
            # compute_metrics=_compute_metrics,
        )

        tune_config = {
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "num_train_epochs": tune.choice([2, 3, 4, 5]),
            "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
        }

        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="eval_acc",
            mode="max",
            perturbation_interval=1,
            hyperparam_mutations={
                "weight_decay": tune.uniform(0.0, 0.3),
                "learning_rate": tune.uniform(1e-5, 5e-5),
                "per_device_train_batch_size": [16, 32, 64],
            },
        )
        reporter = CLIReporter(
            parameter_columns={
                "weight_decay": "w_decay",
                "learning_rate": "lr",
                "per_device_train_batch_size": "train_bs/gpu",
                "num_train_epochs": "num_epochs",
            },
            metric_columns=["eval_acc", "eval_loss", "epoch", "training_iteration"],
        )

        ray_default = {
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            "num_train_epochs": tune.choice(list(range(1, 6))),
            "seed": tune.uniform(1, 40),
            "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
        }
        grid = {
            "per_gpu_batch_size": [16, 32],
            "learning_rate": [2e-5, 3e-5, 5e-5],
            "num_epochs": [2, 3, 4]
        }

        bayesian = {
          "per_gpu_batch_size": (16, 64),
          "weight_decay": (0, 0.3),
          "learning_rate": (1e-5, 5e-5),
          "warmup_steps": (0, 500),
          "num_epochs": (2, 5)
        }
        population = {
          "per_gpu_batch_size": [16, 32, 64],
          "weight_decay": (0, 0.3),
          "learning_rate": (1e-5, 5e-5),
          "num_epochs": [2, 3, 4, 5]
        }

        best_run = trainer.hyperparameter_search(
            n_trials=10,
            direction="maximize",
            backend="ray",
            resources_per_trial={"cpu": hyperparameter_cpus, "gpu": 0},
            stop={"training_iteration": 1} if smoke_test else None,
            progress_reporter=reporter,
            # scheduler=scheduler,
            # local_dir="".join([args.output_dir, "./ray_results/"]),
            log_to_file=True,
            )
        print("\nTUNED:\n", best_run.hyperparameters, "\n")
        tuned_path = "".join([args.output_dir, "/tuned_hyperparameters.json"])
        with open(tuned_path, 'w', encoding='utf-8') as f:
            json.dump(best_run.hyperparameters, f, ensure_ascii=False, indent=4)
        # take optimal parameters for model
        # for n, v in best_run.hyperparameters.items():
        #     setattr(trainer.args, n, v)
        warn_tune = "".join([
            "It is not possible to pass tuned hyperparameters directly due to \
            a bug in how the random number generation seed is handled! \
            The tuned hyperparameters are output to:\n", tuned_path,
            "\nand you can pass these to the trainer with --hyperparameter_file"
        ])
        warn(warn_tune)
        return

    # TUNED:
    _tuned = [
        ('learning_rate', 5.61151641533451e-06),
        ('num_train_epochs', 5),
        ('seed', 8.153956804780389),
        ('per_device_train_batch_size', 64)
        ]

    print(trainer)
    train = trainer.train()
    print(train)
    trainer.save_model()

    # evaluate model using desired metrics
    eval = trainer.evaluate()
    print(eval)

if __name__ == "__main__":
    main()
