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
                        help='provide a torch.bin file of hyperparameters. \
                        NOTE: if given, this overrides all other arguments!')
    parser.add_argument('-e', '--entity_name', type=str, default="",
                        help='provide wandb team name (if available). \
                        NOTE: has no effect if wandb is disabled.')
    parser.add_argument('-p', '--project_name', type=str, default="",
                        help='provide wandb project name (if available). \
                        NOTE: has no effect if wandb is disabled.')
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
    if wandb_state is True:
        wandb.login()
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
    def _compute_metrics(eval_preds):
        metrics = dict()
        accuracy_metric = load_metric('accuracy')
        precision_metric = load_metric('precision')
        recall_metric = load_metric('recall')
        f1_metric = load_metric('f1')
        logits = eval_preds.predictions
        labels = eval_preds.label_ids
        preds = np.argmax(logits, axis=-1)
        wandb.log({"roc_curve" : wandb.plot.roc_curve(
            labels, preds, labels=labels
            )})
        wandb.log({"pr" : wandb.plot.pr_curve(
            labels, preds, labels=labels, classes_to_plot=None
            )})
        # wandb.log({'heatmap_with_text': wandb.plots.HeatMap(
        #     x_labels, y_labels, matrix_values, show_text=True
        #     )})
        # wandb.log({'heatmap_no_text': wandb.plots.HeatMap(
        #     x_labels, y_labels, matrix_values, show_text=False
        #     )})
        # wandb.sklearn.plot_confusion_matrix(y_test, y_pred, nb.classes_)
        metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
        metrics.update(precision_metric.compute(predictions=preds, references=labels, average='weighted'))
        metrics.update(recall_metric.compute(predictions=preds, references=labels, average='weighted'))
        metrics.update(f1_metric.compute(predictions=preds, references=labels, average='weighted'))
        return metrics

    if os.path.exists(hyperparameter_file):
        warn("".join([
            "Loading existing hyperparameters from: ", hyperparameter_file,
            "This overrides all other command line arguments except output_dir!"
            ]))
        args_train = torch.load(hyperparameter_file)
        assert type(args_train) == transformers.training_args.TrainingArguments,
            "hyperparameter file must be a pytorch formatted training_args.bin!"
        args_train.output_dir = args.output_dir
    else:
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
            fp16=fp16, #True,
            push_to_hub=args.push_to_hub, #False,
            label_names=args.label_names, #["labels"],
            report_to=args.report_to,
            run_name=args.run_name,
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
    wandb.save(os.path.join(wandb.run.dir, "pytorch_model.bin"))
    trainer.save_model(model_out)

    # evaluate model using desired metrics
    eval = trainer.evaluate()
    print(eval)

if __name__ == "__main__":
    main()
