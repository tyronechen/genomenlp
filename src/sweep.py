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
    parser.add_argument('-c', '--hyperparameter_cpus', type=int, default=1,
                        help='number of cpus for hyperparameter tuning. \
                        NOTE: has no effect if wandb is on (default)')
    parser.add_argument('-w', '--hyperparameter_sweep', type=str, default="",
                        help='run a hyperparameter sweep with config from file')
    parser.add_argument('-n', '--sweep_count', type=int, default=8,
                        help='run n hyperparameter sweeps (DEFAULT: 64) \
                        NOTE: has no effect if wandb sweep is not enabled.')
    parser.add_argument('-e', '--entity_name', type=str, default="",
                        help='provide wandb team name (if available). \
                        NOTE: has no effect if wandb sweep is not enabled.')
    parser.add_argument('-p', '--project_name', type=str, default="",
                        help='provide wandb project name (if available). \
                        NOTE: has no effect if wandb sweep is not enabled.')
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
    hyperparameter_cpus = args.hyperparameter_cpus
    hyperparameter_sweep = args.hyperparameter_sweep
    sweep_count = args.sweep_count
    vocab_size = args.vocab_size
    shuffle = args.no_shuffle
    wandb_state = args.wandb_off
    entity_name = args.entity_name
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
    dataset["train"].features[args.label_names].names = ["NEG", "POS"]
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

    # hyperparameter tuning on 10% of original datasett following:
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
                config=config,
                settings=wandb.Settings(console='off', start_method='fork'),
                entity=args.entity_name
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
        # these two lines dont work for some reason
        # best_run = sweep.best_run()
        # print(best_run)
        metric_opt = sweep_config["metric"]["name"]
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
    else:
        warn("wandb hyperparameter tuning is disabled, using 🤗 tuner.")
        trainer = Trainer(
            model_init=_model_init,
            tokenizer=tokeniser,
            args=args_train,
            train_dataset=dataset["train"],#.shard(index=1, num_shards=100),
            eval_dataset=dataset["valid"],
            data_collator=data_collator,
            # disable_tqdm=args.disable_tqdm,
            # compute_metrics=_compute_metrics,
        )
        reporter = CLIReporter(
            parameter_columns={
                "weight_decay": "w_decay",
                "learning_rate": "lr",
                "per_device_train_batch_size": "train_bs/gpu",
                "num_train_epochs": "num_epochs",
            },
            metric_columns=["eval_acc", "eval_loss",
                            "epoch", "training_iteration"],
        )
        best_run = trainer.hyperparameter_search(
            n_trials=10,
            direction="maximize",
            backend="ray",
            resources_per_trial={"cpu": hyperparameter_cpus, "gpu": 0},
            # stop={"training_iteration": 1} if smoke_test else None,
            progress_reporter=reporter,
            # scheduler=scheduler,
            # local_dir="".join([args.output_dir, "./ray_results/"]),
            log_to_file=True,
            )

        print("\nTUNED:\n", best_run.hyperparameters, "\n")
        tuned_path = "".join(
            [args.output_dir, "/tuned_hyperparameters.json"]
            )
        with open(tuned_path, 'w', encoding='utf-8') as f:
            json.dump(
                best_run.hyperparameters, f, ensure_ascii=False, indent=4
                )
        print("\nHYPERPARAMETER SWEEP END")
        warn_tune = "".join([
            "It is not possible to pass tuned hyperparameters \
            directly due to a bug in how the random number generation \
            seed is handled! The tuned hyperparameters are output to:",
            "\n", tuned_path,
        ])
        warn(warn_tune)


if __name__ == "__main__":
    main()
