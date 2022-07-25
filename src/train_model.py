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
from transformers import AutoModelForSequenceClassification, DistilBertConfig, \
    DistilBertForSequenceClassification, HfArgumentParser, \
    PreTrainedTokenizerFast, Trainer, TrainingArguments, set_seed
from transformers.training_args import ParallelMode
# import nevergrad as ng
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import download_data, \
    build_compute_metrics_fn
import wandb

# def __todo_in_future():
# from ray.tune.schedulers import PopulationBasedTraining
# from ray.tune.suggest.basic_variant import BasicVariantGenerator
# from ray.tune.suggest.bayesopt import BayesOptSearch
# from ray.tune.suggest.bohb import TuneBOHB
# from ray.tune.suggest.dragonfly import DragonflySearch
# # from ray.tune.suggest.flaml import BlendSearch, CFO
# from ray.tune.suggest.hebo import HEBOSearch
# from ray.tune.suggest.hyperopt import HyperOptSearch
# from ray.tune.suggest.nevergrad import NevergradSearch
# from ray.tune.suggest.optuna import OptunaSearch
# from ray.tune.suggest.sigopt import SigOptSearch
# from ray.tune.suggest.skopt import SkOptSearch
# from ray.tune.suggest.zoopt import ZOOptSearch
# from ray.tune.suggest import Repeater, ConcurrencyLimiter
#     # TODO:
#     # algorithm
#     # small model
#     #   random search
#     #   grid search
#     # big model + small hyperparams
#     #   bayesopt
#     #   dragonfly
#     #   ax
#     #   population
#     config = {
#         "width": tune.uniform(0, 20),
#         "height": tune.uniform(-100, 100)
#     }
#     bayesopt = BayesOptSearch(metric="mean_loss", mode="min")
#     tune.run(my_func, config=config, search_alg=bayesopt)
#
#     config = {
#         "width": tune.uniform(0, 20),
#         "height": tune.uniform(-100, 100),
#         "activation": tune.choice(["relu", "tanh"])
#     }
#     algo = TuneBOHB(metric="mean_loss", mode="min")
#     bohb = HyperBandForBOHB(
#         time_attr="training_iteration",
#         metric="mean_loss",
#         mode="min",
#         max_t=100)
#     run(my_trainable, config=config, scheduler=bohb, search_alg=algo)
#
#     config = {
#         "LiNO3_vol": tune.uniform(0, 7),
#         "Li2SO4_vol": tune.uniform(0, 7),
#         "NaClO4_vol": tune.uniform(0, 7)
#     }
#     df_search = DragonflySearch(
#         optimizer="bandit",
#         domain="euclidean",
#         metric="objective",
#         mode="max")
#     tune.run(my_func, config=config, search_alg=df_search)
#
#     config = {
#         "width": tune.uniform(0, 20),
#         "height": tune.uniform(-100, 100)
#     }
#     hebo = HEBOSearch(metric="mean_loss", mode="min")
#     tune.run(my_func, config=config, search_alg=hebo)
#
#     config = {
#         'width': tune.uniform(0, 20),
#         'height': tune.uniform(-100, 100),
#         'activation': tune.choice(["relu", "tanh"])
#     }
#     current_best_params = [{
#         'width': 10,
#         'height': 0,
#         'activation': "relu",
#     }]
#     hyperopt_search = HyperOptSearch(
#         metric="mean_loss", mode="min",
#         points_to_evaluate=current_best_params)
#     tune.run(trainable, config=config, search_alg=hyperopt_search)
#
#     config = {
#         "width": tune.uniform(0, 20),
#         "height": tune.uniform(-100, 100),
#         "activation": tune.choice(["relu", "tanh"])
#     }
#     current_best_params = [{
#         "width": 10,
#         "height": 0,
#         "activation": "relu",
#     }]
#     ng_search = NevergradSearch(
#         optimizer=ng.optimizers.OnePlusOne,
#         metric="mean_loss",
#         mode="min",
#         points_to_evaluate=current_best_params)
#     run(my_trainable, config=config, search_alg=ng_search)
#
#     config = {
#         "a": tune.uniform(6, 8),
#         "b": tune.loguniform(1e-4, 1e-2)
#     }
#     optuna_search = OptunaSearch(
#         metric="loss",
#         mode="min")
#     tune.run(trainable, config=config, search_alg=optuna_search)
#
#     config = {
#         "a": tune.uniform(6, 8),
#         "b": tune.loguniform(1e-4, 1e-2)
#     }
#     optuna_search = OptunaSearch(
#         metric="loss",
#         mode="min")
#     tune.run(trainable, config=config, search_alg=optuna_search)
#
#     config = {
#         "width": tune.uniform(0, 20),
#         "height": tune.uniform(-100, 100)
#     }
#     current_best_params = [
#         {
#             "width": 10,
#             "height": 0,
#         },
#         {
#             "width": 15,
#             "height": -20,
#         }
#     ]
#     skopt_search = SkOptSearch(
#         metric="mean_loss",
#         mode="min",
#         points_to_evaluate=current_best_params)
#     tune.run(my_trainable, config=config, search_alg=skopt_search)
#
#     config = {
#         "iterations": 10,  # evaluation times
#         "width": tune.uniform(-10, 10),
#         "height": tune.uniform(-10, 10)
#     }
#     zoopt_search_config = {
#         "parallel_num": 8,  # how many workers to parallel
#     }
#     zoopt_search = ZOOptSearch(
#         algo="Asracos",  # only support Asracos currently
#         budget=20,  # must match `num_samples` in `tune.run()`.
#         dim_dict=dim_dict,
#         metric="mean_loss",
#         mode="min",
#         **zoopt_search_config
#     )
#
#     tune.run(my_objective,
#         config=config,
#         search_alg=zoopt_search,
#         name="zoopt_search",
#         num_samples=20,
#         stop={"timesteps_total": 10})
#
#     # WARNING: meta-algorithm, do not use with schedulers!
#     # re_search_alg = Repeater(search_alg, repeat=10)
#     # Repeat 2 samples 10 times each.
#     tune.run(trainable, num_samples=20, search_alg=re_search_alg)
#
#     # WARNING: meta-algorithm, useful in less parallel settings!
#     search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)
#     tune.run(trainable, search_alg=search_alg)
#
#     # small hyperparams in general
#     #   BOHB + ASHA
#     #   optuna + ASHA
#     # continuous hyperparams
#     #   Bayesian
#     # categorical hyperparams
#     #   Bayesian
#     #   random search
#     # # scheduler
#     # HyperBand
#     # ASHA
#     # MedianStoppingRule
#     # PopulationBasedTraining
#     # PopulationBasedBandits - gaussian instead of random perturbation
#     # HyperBandForBOHB - what the name implies
#     # FIFOscheduler - runs in submission order
#     # # Shim instantiation
#     # TODO: flaml and cfo (both require strong initial seed)
#
#     smoke_test = True
#     tune_config = {
#         "per_device_train_batch_size": 32,
#         "per_device_eval_batch_size": 32,
#         "num_train_epochs": tune.choice([2, 3, 4, 5]),
#         "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
#     }
#
#     scheduler = PopulationBasedTraining(
#         time_attr="training_iteration",
#         metric="eval_acc",
#         mode="max",
#         perturbation_interval=1,
#         hyperparam_mutations={
#             "weight_decay": tune.uniform(0.0, 0.3),
#             "learning_rate": tune.uniform(1e-5, 5e-5),
#             "per_device_train_batch_size": [16, 32, 64],
#         },
#     )
#     reporter = CLIReporter(
#         parameter_columns={
#             "weight_decay": "w_decay",
#             "learning_rate": "lr",
#             "per_device_train_batch_size": "train_bs/gpu",
#             "num_train_epochs": "num_epochs",
#         },
#         metric_columns=["eval_acc", "eval_loss", "epoch", "training_iteration"],
#     )
#
#     ray_default = {
#         "learning_rate": tune.loguniform(1e-6, 1e-4),
#         "num_train_epochs": tune.choice(list(range(1, 6))),
#         "seed": tune.uniform(1, 40),
#         "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
#     }
#
#     trainer.hyperparameter_search(
#         hp_space=lambda _: tune_config,
#         backend="ray",
#         n_trials=num_samples,
#         resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
#         scheduler=scheduler,
#         keep_checkpoints_num=1,
#         checkpoint_score_attr="training_iteration",
#         stop={"training_iteration": 1} if smoke_test else None,
#         progress_reporter=reporter,
#         local_dir="~/ray_results/",
#         name="tune_transformer_pbt",
#         log_to_file=True,
#     )
#
#     tune_config = {
#         "per_device_train_batch_size": 32,
#         "per_device_eval_batch_size": 32,
#         "num_train_epochs": tune.choice([2, 3, 4, 5]),
#         "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
#     }
#     search_alg = HyperOptSearch()
#     search_alg = BasicVariantGenerator(points_to_evaluate=[
#             {"a": 2, "b": 2},
#             {"a": 1},
#             {"b": 2}
#         ])
#     experiment_1 = tune.run(
#         trainable,
#         search_alg=search_alg
#         )
#
#     search_alg.save("./my-checkpoint.pkl")
#     # Restore the saved state onto another search algorithm
#
#     search_alg2 = HyperOptSearch()
#     # search_alg2.restore("./my-checkpoint.pkl")
#
#     experiment_2 = tune.run(
#         trainable,
#         search_alg=search_alg2
#         )
#     pass

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
    parser.add_argument('-s', '--vocab_size', type=int, default=32000,
                        help='vocabulary size for model configuration')
    parser.add_argument('-c', '--hyperparameter_cpus', type=int, default=1,
                        help='number of cpus for hyperparameter tuning. \
                        NOTE: has no effect if --hyperparameter tune is False')
    parser.add_argument('-x', '--hyperparameter_tune', type=bool, default=False,
                        help='enable or disable hyperparameter tuning')
    parser.add_argument('-w', '--hyperparameter_sweep', type=str, default="",
                        help='run a hyperparameter sweep with config from file')
    parser.add_argument('-f', '--hyperparameter_file', type=str, default="",
                        help='provide a json file of hyperparameters. \
                        NOTE: if given, this overrides --hyperparameter_tune!')
    parser.add_argument('-n', '--sweep_count', type=int, default=64,
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
    hyperparameter_sweep = args.hyperparameter_sweep
    sweep_count = args.sweep_count
    vocab_size = args.vocab_size
    shuffle = args.no_shuffle
    wandb_state = args.wandb_off
    entity_name = args.entity_name
    project_name = args.project_name
    if wandb_state is True:
        wandb.login()

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
    dataset = dataset.remove_columns("token_type_ids")
    print("\nSAMPLE DATASET ENTRY:\n", dataset["train"][0], "\n")

    col_torch = ['input_ids', 'attention_mask', 'labels']
    # col_torch = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    print(dataset)
    dataset.set_format(type='torch', columns=col_torch)
    dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=1)
    print("\nSAMPLE PYTORCH FORMATTED ENTRY:\n", next(iter(dataloader)))

    config = DistilBertConfig(vocab_size=vocab_size, num_labels=2)
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
        metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
        metrics.update(precision_metric.compute(predictions=preds, references=labels, average='weighted'))
        metrics.update(recall_metric.compute(predictions=preds, references=labels, average='weighted'))
        metrics.update(f1_metric.compute(predictions=preds, references=labels, average='weighted'))
        return metrics

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
            train_dataset=dataset["train"],#.shard(index=1, num_shards=100),
            eval_dataset=dataset["valid"],
            # disable_tqdm=args.disable_tqdm,
            # compute_metrics=_compute_metrics,
        )

        # if hyperparameter sweep is provided or set to True,
        if hyperparameter_sweep != "":
            if wandb_state == True:
                if os.path.exists(hyperparameter_sweep):
                    print(" ".join(["Loading sweep hyperparameters from",
                                   hyperparameter_sweep]))
                    with open(hyperparameter_sweep, 'r') as infile:
                        sweep_config = json.load(infile)
                else:
                    print("No sweep hyperparameters provided, do random search")
                    # method
                    sweep_config = {
                        'name': 'random',
                        'method': 'random',
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

                sweep_id = wandb.sweep(sweep_config, project=project_name)
                # function needs to be defined internally to access namespace
                def _sweep_wandb(config: dict):
                    with wandb.init(
                        config=sweep_config,
                        settings=wandb.Settings(console='off', start_method='fork'),
                        entity=args.entity_name
                        ):
                        # set sweep configuration
                        config = wandb.config

                        # set training arguments
                        training_args = TrainingArguments(
                            output_dir="/".join([args.output_dir, sweep_config["name"]]),
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
                            fp16=False,
                            bf16=False,
                        )
                        # define training loop
                        trainer = Trainer(
                            model_init=_model_init,
                            args=training_args,
                            tokenizer=tokeniser,
                            train_dataset=dataset['train'],
                            eval_dataset=dataset['valid'],#.shard(index=1, num_shards=10),
                            compute_metrics=_compute_metrics
                        )
                        # start training loop
                        trainer.train()
                    return

                # allow passing of argument with variable outside namespace
                wandb_train_func = functools.partial(_sweep_wandb, sweep_config)
                wandb.agent(sweep_id,
                            function=wandb_train_func,
                            count=sweep_count)
                wandb.finish()
                swept = "".join([args.entity_name, sweep_config["name"], sweep_id])
                print("Sweep end:\n", swept)
                api = wandb.Api()
                sweep = api.sweep(swept)
                best_run = sweep.best_run()
                print(best_run)
                runs = sorted(
                    sweep.runs,
                    key=lambda run: run.summary.get("val_acc", 0),
                    reverse=True
                    )
                val_acc = runs[0].summary.get("val_acc", 0)
                print(f"Best run {runs[0].name} with {val_acc}% valn accuracy")
                runs[0].file("model.h5").download(replace=True)
                print("Best model saved to model-best.h5")
                print("\nTUNED:\n", best_run.hyperparameters, "\n")
                tuned_path = "".join([args.output_dir, "/tuned_hyperparameters.json"])
                with open(tuned_path, 'w', encoding='utf-8') as f:
                    json.dump(best_run.hyperparameters, f, ensure_ascii=False, indent=4)
        else:
            warn("wandb hyperparameter tuning is disabled, using 🤗 tuner.")
            reporter = CLIReporter(
                parameter_columns={
                    "weight_decay": "w_decay",
                    "learning_rate": "lr",
                    "per_device_train_batch_size": "train_bs/gpu",
                    "num_train_epochs": "num_epochs",
                },
                metric_columns=["eval_acc", "eval_loss", "epoch", "training_iteration"],
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
            tuned_path = "".join([args.output_dir, "/tuned_hyperparameters.json"])
            with open(tuned_path, 'w', encoding='utf-8') as f:
                json.dump(best_run.hyperparameters, f, ensure_ascii=False, indent=4)
            warn_tune = "".join([
                "It is not possible to pass tuned hyperparameters \
                directly due to a bug in how the random number generation \
                seed is handled! The tuned hyperparameters are output to:",
                "\n", tuned_path, "\nand you can pass these to the trainer \
                with --hyperparameter_file"
            ])
            warn(warn_tune)
        print("Hyperparameter tune end")
        return

    print(trainer)
    train = trainer.train()
    print(train)
    trainer.save_model()

    # evaluate model using desired metrics
    eval = trainer.evaluate()
    print(eval)

if __name__ == "__main__":
    main()
