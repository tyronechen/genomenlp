#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# generic tools
import itertools
import json
import os
from math import ceil
from random import choices, shuffle
from warnings import warn
from datasets import Dataset, DatasetDict, load_metric
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import screed
from sklearn import metrics
from sklearn.inspection import permutation_importance
from tqdm import tqdm
import transformers
from transformers import PreTrainedTokenizerFast, AutoModel, TrainingArguments
# import weightwatcher as ww

# resolves issue of truncated csv
np.set_printoptions(threshold=np.inf)

def _init_sp_tokeniser(vocab=None, weight=-1):
    """Helper function to generate SP-like formatted tokeniser from k-mers"""
    tokeniser = dict()
    tokeniser["version"] = "1.0"
    tokeniser["truncation"] = None
    tokeniser["padding"] = None
    tokeniser["added_tokens"] = [
        {
            'id': 0,
            'content': '<s>',
            'single_word': False,
            'lstrip': False,
            'rstrip': False,
            'normalized': False,
            'special': True
        },
        {
            'id': 1,
            'content': '</s>',
            'single_word': False,
            'lstrip': False,
            'rstrip': False,
            'normalized': False,
            'special': True
        },
        {
            'id': 2,
            'content': '<unk>',
            'single_word': False,
            'lstrip': False,
            'rstrip': False,
            'normalized': False,
            'special': True
        },
        {
            'id': 3,
            'content': '<pad>',
            'single_word': False,
            'lstrip': False,
            'rstrip': False,
            'normalized': False,
            'special': True
        },
        {
            'id': 4,
            'content': '<mask>',
            'single_word': False,
            'lstrip': False,
            'rstrip': False,
            'normalized': False,
            'special': True
        }
    ]
    tokeniser["normalizer"] = {
        'type': 'Sequence',
        'normalizers': [
            {'type': 'Nmt'},
            {'type': 'NFKC'},
            {'type': 'Replace', 'pattern': {'Regex': ' {2,}'}, 'content': ' '}
            ]
        }
    tokeniser["pre_tokenizer"] = {
        'type': 'Metaspace', 'replacement': '▁', 'add_prefix_space': True
        }
    tokeniser["post_processor"] = None
    tokeniser["decoder"] = tokeniser["pre_tokenizer"]
    model = dict()
    model["type"] = "Unigram"
    model["unk_id"] = 2
    model["vocab"] = [
        ['<s>', 0.0],
        ['</s>', 0.0],
        ['<unk>', 0.0],
        ['<pad>', 0.0],
        ['<mask>', 0.0]
        ] + [[i, weight] for i in vocab]
    tokeniser["model"] = model
    return tokeniser

def _init_sp_tokeniser_variable_weights(vocab_weight=None):
    """Helper function to generate SP-like formatted tokeniser from k-mers with variable weights"""
    tokeniser = dict()
    tokeniser["version"] = "1.0"
    tokeniser["truncation"] = None
    tokeniser["padding"] = None
    tokeniser["added_tokens"] = [
        {
            'id': 0,
            'content': '<s>',
            'single_word': False,
            'lstrip': False,
            'rstrip': False,
            'normalized': False,
            'special': True
        },
        {
            'id': 1,
            'content': '</s>',
            'single_word': False,
            'lstrip': False,
            'rstrip': False,
            'normalized': False,
            'special': True
        },
        {
            'id': 2,
            'content': '<unk>',
            'single_word': False,
            'lstrip': False,
            'rstrip': False,
            'normalized': False,
            'special': True
        },
        {
            'id': 3,
            'content': '<pad>',
            'single_word': False,
            'lstrip': False,
            'rstrip': False,
            'normalized': False,
            'special': True
        },
        {
            'id': 4,
            'content': '<mask>',
            'single_word': False,
            'lstrip': False,
            'rstrip': False,
            'normalized': False,
            'special': True
        }
    ]
    tokeniser["normalizer"] = {
        'type': 'Sequence',
        'normalizers': [
            {'type': 'Nmt'},
            {'type': 'NFKC'},
            {'type': 'Replace', 'pattern': {'Regex': ' {2,}'}, 'content': ' '}
            ]
        }
    tokeniser["pre_tokenizer"] = {
        'type': 'Metaspace', 'replacement': '▁', 'add_prefix_space': True
        }
    tokeniser["post_processor"] = None
    tokeniser["decoder"] = tokeniser["pre_tokenizer"]
    model = dict()
    model["type"] = "Unigram"
    model["unk_id"] = 2
    model["vocab"] = [
        ['<s>', 0.0],
        ['</s>', 0.0],
        ['<unk>', 0.0],
        ['<pad>', 0.0],
        ['<mask>', 0.0]
        ] + [[k, v] for k, v in tqdm(vocab_weight.items(), desc="Generating tokens:")]
    tokeniser["model"] = model
    return tokeniser

def _compute_metrics(eval_preds):
    """Compute metrics during the training run using transformers and wandb API.
    FOR REFERENCE ONLY. DO NOT IMPORT OR USE DIRECTLY, FUNNY THINGS WILL HAPPEN.

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

def _cite_me(is_tex: bool=False):
    """Print the citation for this paper before and after each run"""
    # to be added once the manuscript is published
    if is_tex:
        manuscript = """
        Cite our manuscript here:

            @article{chen2023genomicbert,
                title={genomicBERT and data-free deep-learning model evaluation},
                author={Chen, Tyrone and Tyagi, Navya and Chauhan, Sarthak and Peleg, Anton Y and Tyagi, Sonika},
                journal={bioRxiv},
                month={jun},
                pages={2023--05},
                year={2023},
                publisher={Cold Spring Harbor Laboratory},
                doi={10.1101/2023.05.31.542682},
                url={https://doi.org/10.1101/2023.05.31.542682}
            }
        """
        zenodo = """
        Cite our software here:

            @software{tyrone_chen_2023_8135591,
                author       = {Tyrone Chen and
                                Navya Tyagi and
                                Sarthak Chauhan and
                                Anton Y. Peleg and
                                Sonika Tyagi},
                title        = {{genomicBERT and data-free deep-learning model 
                                evaluation}},
                month        = jul,
                year         = 2023,
                publisher    = {Zenodo},
                version      = {latest},
                doi          = {10.5281/zenodo.8135590},
                url          = {https://doi.org/10.5281/zenodo.8135590}
            }
        """
    else:
        manuscript = """
        Cite our manuscript here:

            Chen, T., Tyagi, N., Chauhan, S., Peleg, A.Y. and Tyagi, S., 2023. genomicBERT and data-free deep-learning model evaluation. bioRxiv, pp.2023-05.
        """
        zenodo = """
        Cite our software here:

            Chen, Tyrone, Tyagi, Navya, Chauhan, Sarthak, Peleg, Anton Y., & Tyagi, Sonika. (2023). genomicBERT and data-free deep-learning model evaluation (latest). Zenodo. https://doi.org/10.5281/zenodo.8135590
        """

    print(manuscript)
    print(zenodo)
    pass

def get_feature_importance_mdi(clf, features, model_type, show_features: int=50,
                               output_dir: str=".") -> pd.Series:
    """Calculate feature importance by Gini scores. This is more effective when
    there are fewer classes. See also :py:func:`get_feature_importance_per`.

    Args:
        clf (sklearn.ensemble): a trained sklearn tree-like model.
        features (np.ndarray): the output of `get_feature_names_out`.
        model_type (str): Random Forest "rf" or XGBoost "xg".
        show_features (int): number of features to plot (text export unaffected)
        output_dir (str): figure and list of feature importances go here.

    Returns:
        pd.Series:

        pandas Series object with feature importance scores mapped to features.
    """
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    importances = importances[order]
    features = np.array(features[order])

    mdi_scores = pd.Series(importances, index=features)
    mdi_scores.to_csv("/".join([output_dir, "mdi_scores.tsv"]), sep="\t")

    fig, ax = plt.subplots(figsize=(8, 8))
    if model_type == "rf":
        std = np.std(
            [x.feature_importances_ for x in clf.estimators_], axis=0
            )[order]
        mdi_scores[:show_features].plot.barh(yerr=std[:show_features], ax=ax)
    else:
        mdi_scores[:show_features].plot.barh(ax=ax)

    ax.set_title("Feature importances using MDI")
    ax.set_xlabel("Mean decrease in impurity (Gini importance)")
    fig.tight_layout()
    fig.savefig("/".join([output_dir, "mdi_scores.pdf"]), dpi=300)
    plt.close()
    return mdi_scores

def get_feature_importance_per(clf, x_test, y_test, features, model_type,
                               show_features: int=50, output_dir: str=".",
                               n_repeats: int=10, n_jobs: int=1) -> pd.Series:
    """Calculate feature importance by permutation. This tests feature
    importance in the context of the model only. See also
    :py:func:`get_feature_importance_mdi`.

    Args:
        clf (sklearn.ensemble): a trained sklearn tree-like model.
        x_test (np.ndarray): test data.
        y_test (np.ndarray): test labels.
        features (np.ndarray): the output of `get_feature_names_out`.
        show_features (int): number of features to plot (text export unaffected)
        output_dir (str): figure and list of feature importances go here.
        n_repeats (int): number of repeats for the permutation to run.
        n_jobs (int): number of threads for the permutation to run on.

    Returns:
        pd.Series:

        pandas Series object with feature importance scores mapped to features.
    """
    per_scores = permutation_importance(
        clf, x_test, y_test, n_repeats=n_repeats, n_jobs=n_jobs
        )
    importances = per_scores.importances_mean
    order = np.argsort(importances)[::-1]
    importances = importances[order]
    features = np.array(features[order])

    per_scores = pd.Series(importances, index=features)
    per_scores.to_csv("/".join([output_dir, "per_scores.tsv"]), sep="\t")

    fig, ax = plt.subplots(figsize=(8, 8))

    per_scores[:show_features].plot.barh(ax=ax)

    ax.set_title("Feature importances using permutation")
    ax.set_xlabel("Mean decrease in accuracy")
    fig.tight_layout()
    fig.savefig("/".join([output_dir, "per_scores.pdf"]), dpi=300)
    plt.close()
    return per_scores

def build_kmers(sequence: str, ksize: int) -> str:
    """Generator that takes a fasta sequence and kmer size to return kmers

    Args:
        sequence (str): an instance of a dna sequence.
        ksize (int): size of the k-mer

    Returns:
        str:

        Individual k-mers from the input sequence. If you want to control the
        sliding window size, you can slice the resulting output of this, e.g.

            i for i in build_kmers('ACTGACTGA', 3)]
            ['ACG', 'CGT', 'GTA', 'TAC', 'ACG', 'CGT', 'GTA']
            i for i in build_kmers('ACTGACTGA', 3)][::3]
            ['ACG', 'GAC', 'GTA']
    """
    for i in range(len(sequence) - ksize + 1):
        yield sequence[i:i + ksize]

def calculate_auc(run, group_name=None):
    """Calculate AUC for a wandb run. This assumes you logged a ROC curve.

    Args:
        eval_preds (wandb.Run): an instance of a `wandb.Run`.
        group_name (str): a label for the specified group name

    Returns:
        pandas.DataFrame:

        A `pandas.DataFrame` containing AUC scores per class.
    """
    tables = [i for i in run.logged_artifacts() if i.type=="run_table"]
    versions = [i.version for i in run.logged_artifacts()]
    names = [i.name for i in run.logged_artifacts()]
    data = pd.DataFrame(tuple(zip(versions, tables, names)))
    data = data[data[2].str.contains("roc_curve_table")]
    data["version"] = data[0].apply(lambda x: int(x[1:]))
    table = data[data["version"] == data["version"].max()][1].tolist()[0]
    outfile_path = table.download()

    infile_path = "/".join([outfile_path, "roc_curve_table.table.json"])
    with open(infile_path, mode="r") as i:
        j = json.load(i)
        data = pd.DataFrame(j["data"])
        auc = data.groupby(0).apply(lambda x: metrics.auc(x[1], x[2]))
        auc = pd.DataFrame(auc)
        auc.index.name = None
        auc.reset_index(inplace=True)
        auc.columns = ["class", "auc"]
        auc["run_id"] = run.id
        if group_name != None:
            auc["group_name"] = group_name
    return auc

def get_run_metrics(runs, group_name=None):
    """Get metrics for the specified runs as a `pandas.DataFrame`

    This does not directly obtain the runs, you will need to call `wandb.Api`
    first and specify the runs you want before passing them into here.

    Args:
        runs (wandb.Api.runs): a `wandb.Api.runs()` object
        group_name (str): a label for the specified group name

    Returns:
        pandas.DataFrame:

        Writes the metrics obtained from `wandb.Api.runs` directly to disk.
    """

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
    data = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })
    if group_name != None:
        data["group_name"] = group_name
    return data


def load_args_json(args_json: str):
    """Helper function to load a `json` file into TrainingArguments

    Loads a `json` file of arguments into a
    `transformers.training_args.TrainingArguments` object.

    Args:
        args_json (str): Path to `json` file with training arguments

    Returns:
        transformers.training_args.TrainingArguments:

        For more information please refer to the huggingface documentation
        directly: https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/trainer#transformers.TrainingArguments
    """
    with open(hyperparameter_file, mode="r") as infile:
        args_json = json.load(infile)

    args_train = TrainingArguments(
        output_dir=args_json["output_dir"],
        overwrite_output_dir=args_json["overwrite_output_dir"],
        do_train=args_json["do_train"],
        do_eval=args_json["do_eval"],
        do_predict=args_json["do_predict"],
        evaluation_strategy=args_json["evaluation_strategy"],
        prediction_loss_only=args_json["prediction_loss_only"],
        per_device_train_batch_size=args_json["per_device_train_batch_size"],
        per_device_eval_batch_size=args_json["per_device_eval_batch_size"],
        per_gpu_train_batch_size=args_json["per_gpu_train_batch_size"],
        per_gpu_eval_batch_size=args_json["per_gpu_eval_batch_size"],
        gradient_accumulation_steps=args_json["gradient_accumulation_steps"],
        eval_accumulation_steps=args_json["eval_accumulation_steps"],
        eval_delay=args_json["eval_delay"],
        learning_rate=args_json["learning_rate"],
        weight_decay=args_json["weight_decay"],
        adam_beta1=args_json["adam_beta1"],
        adam_beta2=args_json["adam_beta2"],
        adam_epsilon=args_json["adam_epsilon"],
        max_grad_norm=args_json["max_grad_norm"],
        num_train_epochs=args_json["num_train_epochs"],
        max_steps=args_json["max_steps"],
        lr_scheduler_type=args_json["lr_scheduler_type"],
        warmup_ratio=args_json["warmup_ratio"],
        warmup_steps=args_json["warmup_steps"],
        log_level=args_json["log_level"],
        log_level_replica=args_json["log_level_replica"],
        log_on_each_node=args_json["log_on_each_node"],
        logging_dir=args_json["logging_dir"],
        logging_strategy=args_json["logging_strategy"],
        logging_first_step=args_json["logging_first_step"],
        logging_steps=args_json["logging_steps"],
        logging_nan_inf_filter=args_json["logging_nan_inf_filter"],
        save_strategy=args_json["save_strategy"],
        save_steps=args_json["save_steps"],
        save_total_limit=args_json["save_total_limit"],
        save_on_each_node=args_json["save_on_each_node"],
        no_cuda=args_json["no_cuda"],
        use_mps_device=args_json["use_mps_device"],
        seed=args_json["seed"],
        data_seed=args_json["data_seed"],
        jit_mode_eval=args_json["jit_mode_eval"],
        use_ipex=args_json["use_ipex"],
        bf16=args_json["bf16"],
        fp16=args_json["fp16"],
        fp16_opt_level=args_json["fp16_opt_level"],
        half_precision_backend=args_json["half_precision_backend"],
        bf16_full_eval=args_json["bf16_full_eval"],
        fp16_full_eval=args_json["fp16_full_eval"],
        tf32=args_json["tf32"],
        local_rank=args_json["local_rank"],
        xpu_backend=args_json["xpu_backend"],
        tpu_num_cores=args_json["tpu_num_cores"],
        tpu_metrics_debug=args_json["tpu_metrics_debug"],
        debug=args_json["debug"],
        dataloader_drop_last=args_json["dataloader_drop_last"],
        eval_steps=args_json["eval_steps"],
        dataloader_num_workers=args_json["dataloader_num_workers"],
        past_index=args_json["past_index"],
        run_name=args_json["run_name"],
        disable_tqdm=args_json["disable_tqdm"],
        remove_unused_columns=args_json["remove_unused_columns"],
        label_names=args_json["label_names"],
        load_best_model_at_end=args_json["load_best_model_at_end"],
        metric_for_best_model=args_json["metric_for_best_model"],
        greater_is_better=args_json["greater_is_better"],
        ignore_data_skip=args_json["ignore_data_skip"],
        sharded_ddp=args_json["sharded_ddp"],
        fsdp=args_json["fsdp"],
        fsdp_min_num_params=args_json["fsdp_min_num_params"],
        fsdp_transformer_layer_cls_to_wrap=args_json["fsdp_transformer_layer_cls_to_wrap"],
        deepspeed=args_json["deepspeed"],
        label_smoothing_factor=args_json["label_smoothing_factor"],
        optim=args_json["optim"],
        adafactor=args_json["adafactor"],
        group_by_length=args_json["group_by_length"],
        length_column_name=args_json["length_column_name"],
        report_to=args_json["report_to"],
        ddp_find_unused_parameters=args_json["ddp_find_unused_parameters"],
        ddp_bucket_cap_mb=args_json["ddp_bucket_cap_mb"],
        dataloader_pin_memory=args_json["dataloader_pin_memory"],
        skip_memory_metrics=args_json["skip_memory_metrics"],
        use_legacy_prediction_loop=args_json["use_legacy_prediction_loop"],
        push_to_hub=args_json["push_to_hub"],
        resume_from_checkpoint=args_json["resume_from_checkpoint"],
        hub_model_id=args_json["hub_model_id"],
        hub_strategy=args_json["hub_strategy"],
        hub_token=args_json["hub_token"],
        hub_private_repo=args_json["hub_private_repo"],
        gradient_checkpointing=args_json["gradient_checkpointing"],
        include_inputs_for_metrics=args_json["include_inputs_for_metrics"],
        fp16_backend=args_json["fp16_backend"],
        push_to_hub_model_id=args_json["push_to_hub_model_id"],
        push_to_hub_organization=args_json["push_to_hub_organization"],
        push_to_hub_token=args_json["push_to_hub_token"],
        mp_parameters=args_json["mp_parameters"],
        auto_find_batch_size=args_json["auto_find_batch_size"],
        full_determinism=args_json["full_determinism"],
        torchdynamo=args_json["torchdynamo"],
        ray_scope=args_json["ray_scope"],
        ddp_timeout=args_json["ddp_timeout"]
    )
    assert type(args_train) == transformers.training_args.TrainingArguments, \
        "Must be an instance of transformers.training_args.TrainingArguments"
    return args_train

def load_args_cmd(args):
    """Helper function to load a `HfArgumentParser` into `TrainingArguments`

    Loads a `HfArgumentParser` class of arguments into a
    `transformers.training_args.TrainingArguments` object.

    Args:
        args (class): A `HfArgumentParser` object

    Returns:
        transformers.training_args.TrainingArguments:

        For more information please refer to the huggingface documentation
        directly: https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/trainer#transformers.TrainingArguments
    """
    args_train = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        evaluation_strategy=args.evaluation_strategy,
        prediction_loss_only=args.prediction_loss_only,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        per_gpu_train_batch_size=args.per_gpu_train_batch_size,
        per_gpu_eval_batch_size=args.per_gpu_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        eval_delay=args.eval_delay,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        log_level=args.log_level,
        log_level_replica=args.log_level_replica,
        log_on_each_node=args.log_on_each_node,
        logging_dir=args.logging_dir,
        logging_strategy=args.logging_strategy,
        logging_first_step=args.logging_first_step,
        logging_steps=args.logging_steps,
        logging_nan_inf_filter=args.logging_nan_inf_filter,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_on_each_node=args.save_on_each_node,
        no_cuda=args.no_cuda,
        use_mps_device=args.use_mps_device,
        seed=args.seed,
        data_seed=args.data_seed,
        jit_mode_eval=args.jit_mode_eval,
        use_ipex=args.use_ipex,
        bf16=args.bf16,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        half_precision_backend=args.half_precision_backend,
        bf16_full_eval=args.bf16_full_eval,
        fp16_full_eval=args.fp16_full_eval,
        tf32=args.tf32,
        local_rank=args.local_rank,
        xpu_backend=args.xpu_backend,
        tpu_num_cores=args.tpu_num_cores,
        tpu_metrics_debug=args.tpu_metrics_debug,
        debug=args.debug,
        dataloader_drop_last=args.dataloader_drop_last,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        past_index=args.past_index,
        run_name=args.run_name,
        disable_tqdm=args.disable_tqdm,
        remove_unused_columns=args.remove_unused_columns,
        label_names=args.label_names,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        ignore_data_skip=args.ignore_data_skip,
        sharded_ddp=args.sharded_ddp,
        fsdp=args.fsdp,
        fsdp_min_num_params=args.fsdp_min_num_params,
        fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap,
        deepspeed=args.deepspeed,
        label_smoothing_factor=args.label_smoothing_factor,
        optim=args.optim,
        adafactor=args.adafactor,
        group_by_length=args.group_by_length,
        length_column_name=args.length_column_name,
        report_to=args.report_to,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
        dataloader_pin_memory=args.dataloader_pin_memory,
        skip_memory_metrics=args.skip_memory_metrics,
        use_legacy_prediction_loop=args.use_legacy_prediction_loop,
        push_to_hub=args.push_to_hub,
        resume_from_checkpoint=args.resume_from_checkpoint,
        hub_model_id=args.hub_model_id,
        hub_strategy=args.hub_strategy,
        hub_token=args.hub_token,
        hub_private_repo=args.hub_private_repo,
        gradient_checkpointing=args.gradient_checkpointing,
        include_inputs_for_metrics=args.include_inputs_for_metrics,
        fp16_backend=args.fp16_backend,
        push_to_hub_model_id=args.push_to_hub_model_id,
        push_to_hub_organization=args.push_to_hub_organization,
        push_to_hub_token=args.push_to_hub_token,
        mp_parameters=args.mp_parameters,
        auto_find_batch_size=args.auto_find_batch_size,
        full_determinism=args.full_determinism,
        torchdynamo=args.torchdynamo,
        ray_scope=args.ray_scope,
        ddp_timeout=args.ddp_timeout,
    )
    assert type(args_train) == transformers.training_args.TrainingArguments, \
        "Must be an instance of transformers.training_args.TrainingArguments"
    return args_train

def bootstrap_seq(seq: str, block_size: int=2):
    """Take a string and reshuffle it in blocks of N length.

    Shuffles a sequence in the user-defined block size. Joins the
    sequence back together at the end.

    Compare :py:func:`generate_from_freq`.

    Args:
        seq (str): A string of biological sequence data.
        block_size (int): An integer specifying the size of block to shuffle.

    Returns:
        str:

        A reshuffled string of the same length as the original input

        Input: ``ACGT``

        Output: ``GTAC``

        If the reconstructed seq exceeds seq length it will be truncated.
    """
    chunks, chunk_size = len(seq), block_size
    seq = [ seq[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    shuffle(seq)
    return "".join(seq)

def generate_from_freq(seq: str, block_size: int=2,
                       alphabet: list=["A","C","G","T"], offset: float=0.01):
    """Take a string and sample from freq distribution to fill up seq length.

    Compare :py:func:`bootstrap_seq`.

    Args:
        seq (str): A string of biological sequence data
        block_size (int): Size of block to shuffle
        alphabet (list[str]): Biological alphabet present in input sequences
        offset (float): Adding offset avoids 0 division errors in small datasets

    Returns:
        str:

        Resampled sequence with matching frequency distribution of the same
        length as the original input. Frequency distribution is sampled as
        n-length blocks (eg: ``[AA, AC, ..]`` or ``[AAA, AAC, ...]``).

        Input: ``AAAACGT``

        Output: ``ACGTAAA``

        If the reconstructed seq exceeds seq length it will be truncated.
    """
    if len(seq) == 0:
        return
    assert block_size <= len(seq), "Block size exceeds sequence length!"
    to_count = ["".join(i) for i in itertools.product(alphabet, repeat=block_size)]
    count = [str.count(seq, x) + offset for x in to_count]
    freq = dict(zip(to_count, [x / len(count) for x in count]))
    draw_size = ceil(len(seq) / block_size)
    new = choices(list(freq.keys()), weights=freq.values(), k=draw_size)
    return "".join(new)[:len(seq)]

def chunk_text(infile_path: str, outfile_path: str, title: str, labels: str,
               content: str, chunk: int=512):
    """Take a csv-like file of text, process and stream to csv-like file.

    Args:
        infile_path (str): A path to a file containing natural language data
        outfile_path (str): A path to a file containing the output
        title (str): Title of column containing titles (can be an identifier)
        labels (str): Title of column containing labels
        content (str): Title of column containing content
        chunk (int): Chunk the data into seqs of n length (DEFAULT: 512)

    Returns:
        None:

        The file is written directly to disk and the sequences are not returned.

        Input: ``/path/to/infile /path/to/outfile title labels content chunk_size``

        Output: ``None``

        Note that this is specific for natural language data and will not work
        on biological sequences directly (which have specific formatting).
        Here we assume there are the columns: index, title, content, labels.
    """
    text = pd.read_csv(infile_path, chunksize=1, sep=",", header=0, index_col=0)
    for i in text:
        titles = i[title].values[0]
        seq = i[content].values[0]
        label = i[labels].values[0]
        seqs = [seq[i:i + chunk] for i in range(0, len(seq), chunk)]
        subtitles = ["".join([titles, "__PARTIAL_SEQ_CHUNK_", str(i)])
                     for i in range(len(seqs))]
        data = pd.DataFrame({title: subtitles, content: seqs})
        data[labels] = label
        data.to_csv(outfile_path, mode="a", index=False,
                    header=not os.path.exists(outfile_path))

def process_seqs(infile_path: str, outfile_path: str, rc: bool=True, chunk: int=None):
    """Take a file of biological sequences, process and stream to csv-like file.
    Calls :py:func:`reverse_complement`. Used before :py:func:`csv_to_hf`.

    Args:
        infile_path (str): A path to a file containing biological sequence data
        outfile_path (str): A path to a file containing the output
        rc (bool): reverse complement the data (DEFAULT: TRUE)
        chunk (int): chunk the data into seqs of n length (DEFAULT: None)

    Returns:
        None:

        The file is written directly to disk and the sequences are not returned.

        Input: ``/path/to/infile``

        Output: ``None``

        Note that no sequence cleaning is performed, 'N' gets mapped to itself.
        Uppercase is assumed. Does not work on RNA!
    """
    warn("Any commas in fasta headers will be replaced with __!")
    with open(outfile_path, mode="a+") as tmp:
        with screed.open(infile_path) as infile:
            if chunk == None:
                for read in infile:
                    head = read.name.replace(",", "__")
                    seq = read.sequence.upper()
                    tmp.write("".join([head, ",", seq, "\n"]))
                    if rc is True:
                        tmp.write("".join([head, "__RC", ","]))
                        tmp.write("".join([reverse_complement(seq), "\n"]))
            else:
                for read in infile:
                    head = read.name.replace(",", "__")
                    seq = read.sequence.upper()
                    seq = [seq[i:i + chunk] for i in range(0, len(seq), chunk)]
                    for i in range(len(seq)):
                        subhead = "".join([head, "__PARTIAL_SEQ_CHUNK_", str(i)])
                        tmp.write("".join([subhead, ",", seq[i], "\n"]))
                        if rc is True:
                            tmp.write("".join([subhead, "__RC", ","]))
                            tmp.write("".join([reverse_complement(seq[i]), "\n"]))

class _EmbedSeqsKmers(object):
    """this doesnt work correctly, keeps getting the same instance back."""

    def __init__(self, infile_paths: str, ksize: int, slide=1, rc: bool=True, chunk: int=None):
        super(EmbedSeqsKmers, self).__init__()
        self.infile_paths = infile_paths
        self.ksize = ksize
        self.slide = slide
        self.rc = rc
        self.chunk = chunk

    def __iter__(self):
        # return self.embed_seqs_kmers()
        for infile_path in self.infile_paths:
            with screed.open(infile_path) as infile:
                total = len([i for i in infile])
            with screed.open(infile_path) as infile:
                if self.chunk == None:
                    for read in tqdm(infile, total=total):
                        seq = read.sequence.upper()
                        if self.rc is True:
                            seq = "".join([seq, reverse_complement(seq)])
                        yield [i for i in build_kmers(seq, self.ksize)][::self.slide]
                else:
                    for read in tqdm(infile, total=total):
                        seq = read.sequence.upper()
                        seq = [seq[i:i + self.chunk] for i in range(0, len(seq), chunk)]
                        for i in range(len(seq)):
                            if self.rc is True:
                                i = "".join([i, reverse_complement(i)])
                            yield [j for j in build_kmers(i, self.ksize)][::self.slide]

def embed_seqs_kmers(infile_path: str, ksize: int=5, slide: int=1,
                     rc: bool=True, chunk: int=None, outfile_path: str=None):
    """Take a file of biological sequences, process and stream to generator.
    Calls :py:func:`build_kmers` and :py:func:`reverse_complement`.
    Used to generate `word2vec` embeddings.

    Args:
        infile_path (str): A path to a file containing biological sequence data
        ksize (int): size of the k-mer (DEFAULT: 5)
        slide (int): size of the sliding window (DEFAULT: 1)
            If you want no sliding to be performed, set `slide` equal to `ksize`
        rc (bool): reverse complement the data (DEFAULT: TRUE)
        chunk (int): chunk the data into seqs of n length (DEFAULT: None)
        outfile_path (str): A path to outfile (DEFAULT: None)

    Returns:
        list:

        Sequences are returned as a generator object for input into `word2vec`

        Input: ``/path/to/infile``

        Output: ``list``

        Note that no sequence cleaning is performed, 'N' gets mapped to itself.
        Uppercase is assumed. Does not work on RNA!
    """
    with screed.open(infile_path) as infile:
        total = len([i for i in infile])
    with screed.open(infile_path) as infile:
        if chunk == None:
            for read in tqdm(infile, total=total):
                seq = read.sequence.upper()
                if rc is True:
                    seq = "".join([seq, reverse_complement(seq)])
                if outfile_path != None:
                    with open(outfile_path, mode="a+") as tmp:
                        tmp.write(" ".join(seq) + "\n")
                yield [i for i in build_kmers(seq, ksize)][::slide]
        else:
            for read in tqdm(infile, total=total):
                seq = read.sequence.upper()
                seq = [seq[i:i + chunk] for i in range(0, len(seq), chunk)]
                for i in range(len(seq)):
                    if rc is True:
                        i = "".join([i, reverse_complement(i)])
                    if outfile_path != None:
                        with open(outfile_path, mode="a+") as tmp:
                            tmp.write(" ".join(seq) + "\n")
                    yield [j for j in build_kmers(i, ksize)][::slide]

def embed_seqs_sp(infile_path: str, outfile_path: str, chunksize: int=1,
                  tokeniser_path: str=None, special_tokens:
                  list=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                  columns: list=["idx", "feature", "labels", "input_ids",
                  "token_type_ids", "attention_mask", "input_str"],
                  column: str="input_str", labels: str=None):
    """Take a file of SP tokenised sequences, process and stream to generator.
    Used to generate `word2vec` embeddings. See also :py:func:`parse_sp_tokenised`.

    Args:
        infile_path (str): Path to ``csv`` file containing tokenised data.
        outfile_path (str): Path to ``csv`` file containing tokenised data.
        chunksize (int): How many rows of the dataframe to iterate at a time.
        tokeniser_path (str): Path to sequence tokens file
            (from ``SentencePiece``)
        special_tokens (list[str]): Special tokens to substitute for.
            This should match the list of special tokens used in the original
            tokeniser (which defaults to the five special tokens shown here).
        columns (list): List of column headings (in infile_path)
        column (str): The column header with the input_str (to extract tokens)
        labels (str): If specified, return label column (to extract tokens)

    Returns:
        list:

        Sequences are returned as a generator object for input into `word2vec`

        Input: ``/path/to/infile``

        Output: ``list``
    """
    parse_sp_tokenised(
        infile_path=infile_path,
        outfile_path=outfile_path,
        tokeniser_path=tokeniser_path,
        special_tokens=special_tokens,
        chunksize=chunksize,
        columns=columns
    )
    if labels == None:
        for data in tqdm(
            pd.read_csv(outfile_path, index_col=0, chunksize=1),
            desc="Extract SP tokens"
            ):
            sp = data[column].apply(lambda x: x[1:-1].replace("\'", "").split())
            yield sp.iloc[0]
    else:
        for data in tqdm(
            pd.read_csv(outfile_path, index_col=0, chunksize=1),
            desc="Extract SP tokens"
            ):
            data[column] = data[column].apply(
                lambda x: x[1:-1].replace("\'", "").split()
                )
            yield data[[column, labels]].iloc[0].tolist()

def csv_to_hf(infile_neg: str, infile_pos: str, outfile_path: str):
    """Add hf formatting to an existing csv-like file and stream to csv-like file.
    Used downstream of :py:func:`process_seqs`.

    Args:
        infile_neg (str): Path to file containing negative / condition 0 data
        infile_pos (str): Path to file containing positive / condition 1 data
        outfile_path (str): Write huggingface dataset compatible output

    Returns:
        None:

        The file is written directly to disk and the sequences are not returned.

        Input: ``/path/to/infile_one /path/to/infile_two /path/to/output``

        Output: ``None``

        This is intended to be used after :py:func:`process_seqs`. If
        used directly, it may not work as intended as some things are hardcoded.
    """
    with open(outfile_path, mode="a+") as tmp_out:
        tmp_out.write("idx,feature,labels\n")
        seqs = pd.read_csv(infile_neg, chunksize=10000, sep=",", header=None)
        for i in seqs:
            i.rename(columns={0: "idx", 1: "feature"}, inplace=True)
            i["labels"] = "NEGATIVE"
            tmp_out.write(i.to_csv(index=False, header=False, sep=","))
        seqs = pd.read_csv(infile_pos, chunksize=10000, sep=",", header=None)
        for i in seqs:
            i.rename(columns={0: "idx", 1: "feature"}, inplace=True)
            i["labels"] = "POSITIVE"
            tmp_out.write(i.to_csv(index=False, header=False, sep=","))

def reverse_complement(dna: str):
    """Take a nucleic acid string as input and return reverse complement.

    Args:
        dna (str): A string of nucleic acid sequence data.

    Returns:
        str:

        Reverse complemented DNA/RNA string.

        Input: ``ACGT``

        Output: ``TGCA``

        Note that no sequence cleaning is performed, 'N' gets mapped to itself.
        Uppercase is assumed. If U is detected, automatically assume RNA!
        Supports letters YRKMSW. BDHV get converted to N!.
    """
    for i in ["B", "D", "H", "V"]:
        if i in dna:
            warn("B, D, H, V nucleotides detected! These are reverse complemented to N! Also check if your sequences are protein, and if so disable reverse complement.")

    if "U" in dna:
        complement = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A', 'N': 'N', 
                      'Y': 'R', 'R': 'Y', 'K': 'M', 'M': 'K', 'S': 'W', 'W': 'S',
                      'B': 'N', 'D': 'N', 'H': 'N', 'V': 'N'}
    else:
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', 
                      'Y': 'R', 'R': 'Y', 'K': 'M', 'M': 'K', 'S': 'W', 'W': 'S',
                      'B': 'N', 'D': 'N', 'H': 'N', 'V': 'N'}
    return "".join([complement[base] for base in dna[::-1]])

def get_tokens_from_sp(tokeniser_path: str,
                       special_tokens: list=["<s>", "</s>", "<unk>", "<pad>",
                       "<mask>"]):
    """Take path to ``SentencePiece`` tokeniser + special tokens, return tokens

    The input ``tokeniser_path`` is a ``json`` file generated from the
    ``HuggingFace`` implementation of ``SentencePiece``. Compare
    :py:func:`parse_sp_tokenised`.

    Args:
        tokeniser_path (str): Path to sequence tokens file
            (from ``SentencePiece``)
        special_tokens (list[str]): Special tokens to substitute for.
            This should match the list of special tokens used in the original
            tokeniser (which defaults to the five special tokens shown here).

    Returns:
        list:

        A list of cleaned tokens corresponding to variable length k-mers.
    """
    # if we dont specify the special tokens below it will break
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
    return [x.replace("▁", "") for x in list(tokeniser.vocab.keys())]

def parse_sp_tokenised(infile_path: str, outfile_path: str,
                       tokeniser_path: str=None, special_tokens:
                       list=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                       chunksize: int=100, columns: list=["idx", "feature",
                       "labels", "input_ids", "token_type_ids",
                       "attention_mask", "input_str"]):
    """Extract entries tokenised by SentencePiece into a pandas.DataFrame object

    The input ``infile_path`` is a ``csv`` file containing tokenised data as
    positional ordinal encodings. The data should have been tokenised using the
    ``HuggingFace`` implementation of ``SentencePiece``. Writes file to disk.
    Compare :py:func:`get_tokens_from_sp`. See also :py:func:`embed_seqs_sp`.

    Args:
        infile_path (str): Path to ``csv`` file containing tokenised data.
        outfile_path (str): Path to ``csv`` file containing tokenised data.
        tokeniser_path (str): Path to sequence tokens file
            (from ``SentencePiece``)
        special_tokens (list[str]): Special tokens to substitute for.
            This should match the list of special tokens used in the original
            tokeniser (which defaults to the five special tokens shown here).
        chunksize (int): How many rows of the dataframe to iterate at a time.
        columns (list): List of column headings

    Returns:
        None:

        The ``pandas.DataFrame`` contains the contents of the ``csv`` file, but
        numeric columns are correctly formatted as ``numpy.array``. The
        ``remap_file`` argument is useful if you want to extract the k-mers
        directly for use in different workflows.
    """
    # you can only remap if you know the original id: str mappings!
    if tokeniser_path != None:
        # if we dont specify the special tokens below it will break
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
        token_map = {v: k.replace("▁", "")  for k, v in tokeniser.vocab.items()}
        # token_map = {k.replace("▁", ""): v  for k, v in tokeniser.vocab.items()}
    # load the files in chunks to avoid memory issues
    if os.path.exists(outfile_path):
        os.remove(outfile_path)
    with open(outfile_path, mode="a+") as outfile:
        outfile.write(",".join(columns) + "\n")

    # input_str = pd.Series(np.array(
    #     data["input_str"].iloc[0][1:-1].replace("\'", "").split()
    #     ))
    # data["input_ids"] input_ids = str(input_str.apply(
    #     lambda x: np.vectorize(token_map.get)(x)
    #     ).to_list())
    #
    for data in tqdm(
        pd.read_csv(infile_path, index_col=0, chunksize=chunksize),
        desc="Parse SP tokens"):
        data["input_ids"] = data["input_ids"].apply(
            lambda x: np.fromstring(x[1:-1], sep=" ", dtype=int)
            )
        if "token_type_ids" in data:
            data["token_type_ids"] = data["token_type_ids"].apply(
                lambda x: np.fromstring(x[1:-1], sep=" ", dtype=int)
                )
        data["input_str"] = data["input_ids"].apply(
            lambda x: np.vectorize(token_map.get)(x)
        )
        data.to_csv(outfile_path, header=False, mode="a+")

def plot_token_dist(tokeniser_path: str, special_tokens: list=["<s>", "</s>",
                    "<unk>", "<pad>", "<mask>"], outfile_dir: str="./"):
    """Plot distribution of token lengths. Calls :py:func:`get_tokens_from_sp`

    The input ``tokeniser_path`` is a ``json`` file generated from the
    ``HuggingFace`` implementation of ``SentencePiece``.

    Args:
        tokeniser_path (str): Path to sequence tokens file (from ``SentencePiece``)
        special_tokens (list[str]): Special tokens to substitute for
        outfile_dir (str): Path to output plots

    Returns:
        matplotlib.pyplot:

        Token histogram plots are written to ``outfile_dir`` in ``png`` and
        ``pdf`` formats.
    """
    tokens = get_tokens_from_sp(tokeniser_path, special_tokens)
    for special_token in special_tokens:
        tokens.remove(special_token)
    tokens_len = [len(x) for x in tokens]

    for_plot = pd.DataFrame(pd.Series(tokens_len))
    for_plot.columns = ["Selected k-mer lengths (base pairs)"]
    for_plot.index.name = "Quantity (units)"

    hist = for_plot.plot(kind="hist", grid=False, legend=False)
    hist.set_xlabel("Selected k-mer lengths (base pairs)")
    title = "".join(
        ["Selected k-mer length distribution (of ", str(len(tokens_len)), ")"]
        )
    hist.set_title(title)
    plt_out = ["".join(
        [outfile_dir, "kmer_length_histogram.", i]
        ) for i in ["pdf", "png"]]
    [plt.savefig(i, dpi=300) for i in plt_out]
    return hist

def remove_stopwords(dataset: str, column: str=None, highmem: bool=True):
    """Remove English language stopwords from text. Stopwords are obtained from
    ``SpaCy 3.2.4``.

    Args:
        dataset (str): A path to a comma separated ``.csv`` file
        column (str): The name of the column to be cleaned. If no column text is
            provided (*default*), parses all columns. This option is disabled if
            highmem is set to ``False``!
        highmem (bool): If ``True`` (*default*), uses ``pandas`` to operate on
            the file. If ``False``, parses the file line by line, overriding
            column selection!

    Returns:
        str:

        New file path with removed stopwords, named ``dataset.CLEAN``.
        Note that stopwords with leading uppercase are also removed.
        For example "the" and "The" will be treated the same and removed.
        To obtain the stopwords list for English used in this function::

            #!/bin/bash
            python -m spacy download en

            #!/usr/bin/python
            import spacy
            sp = spacy.load('en_core_web_sm')
            stopwords_en = sp.Defaults.stop_words
    """
    # obtained from SpaCy 3.2.4
    stopwords_en = {
        'twelve', 'along', 'for', 'most', '‘d', 'as', 'the', 'in', 'ever',
        'themselves', 'whole', 'here', 'do', 'so', 'elsewhere', 'therefore',
        "'ve", '‘re', 'alone', 'make', 'just', '’ve', 'on', 'eight', 'such',
        'hereupon', "'re", 'whereas', 'is', 'might', 'thereupon', 'yours',
        'because', 'almost', 'how', 'amongst', 'it', 'everything', 'while',
        'anyone', 'whom', 'namely', 'hereafter', 'during', 'quite', "n't",
        'those', 'every', 'beforehand', 'wherein', 'his', 'our', 'beyond', 'no',
        'done', 'six', 'used', 'become', 'within', 'seems', 'have', 'well',
        '’s', 'top', 'keep', 'another', 'none', 'although', 'per', '‘s',
        'which', 'toward', 'four', 'first', 'anyway', '’re', 'her', 'take',
        'am', 'himself', 'too', 'call', 'wherever', 'down', 'into', 'up',
        'unless', 'seemed', 'what', 'thru', 'hundred', 'your', "'m", 'each',
        'does', 'though', 'name', 'hers', 'afterwards', 'some', 'front', 'made',
        'show', 'its', 'perhaps', 'were', 'other', 'than', 'without', 'least',
        'enough', 'by', 'until', 'him', 'from', 'amount', 'say', 'became',
        'yourself', 'throughout', 'about', 'where', 'can', 'former', 'two',
        'rather', 'anywhere', 'off', 'indeed', 'give', 'mostly', 'only', 'back',
        'go', 'put', 'more', 'onto', 'somehow', '’d', '’m', 'ca', 'bottom',
        'cannot', '‘ll', 'we', 'any', 'would', 'nor', 'whither', 'one', 'n’t',
        'herself', 'at', 'everywhere', 'few', 'been', 'between', 'please',
        'below', 'around', 'regarding', 'using', 'across', 'several', 'whereby',
        'fifty', 'less', 'someone', 'get', 'before', 'seeming', 'since',
        'therein', 'myself', 'be', 'sometime', 'to', 'was', 'whenever',
        'latterly', 'three', 'nevertheless', 'whereafter', 'still', 'always',
        'five', 'ourselves', 'serious', 'has', 'should', 'their', 'ours',
        'hence', 'empty', 'n‘t', 'upon', 'formerly', 'them', 'itself', 'all',
        'besides', 'i', 'due', 'under', 'others', 'through', 'whose', 'if',
        'did', 'why', 'mine', 'beside', 'third', 'moreover', 'otherwise', 'via',
        'whoever', "'d", 'or', 'together', 'whence', 'doing', 'thence', 'he',
        'they', 'sometimes', "'s", 'see', 'never', 'against', 'over',
        'whatever', 'next', 'yourselves', 'now', 'part', 'even', 'except',
        'twenty', 'once', 'both', 'thereby', 'ten', 'full', 'anyhow', 'also',
        'noone', 'among', 'are', 'very', '‘ve', 'herein', 'eleven', 'and',
        'after', 'often', 'with', 'nowhere', 'may', 'becoming', 'really', '‘m',
        'my', 'whereupon', 'fifteen', 'same', 'various', 'again', 'nine', 'of',
        'you', 'a', 'behind', 'everyone', '’ll', 'side', 'else', 'further',
        'an', 'either', 'last', "'ll", 'could', 'will', 'must', 'who', 'forty',
        'neither', 'when', 'being', 'move', 'she', 'there', 'us', 'nothing',
        'seem', 'had', 'many', 'that', 'becomes', 'not', 'already', 'towards',
        'this', 'but', 'whether', 'sixty', 'thus', 'these', 'then', 'nobody',
        'anything', 'latter', 're', 'much', 'hereby', 'something', 'me', 'yet',
        'thereafter', 'out', 'meanwhile', 'above', 'however', 'somewhere', 'own'
        }
    # we correctly ignore indexes out of range
    stopwords_en_case = {"".join([i[0].upper(), i[1:]]) for i in stopwords_en}
    stopwords_en = stopwords_en.union(stopwords_en_case)

    outfile_path = ".".join([dataset, "CLEAN"])
    if os.path.exists(outfile_path):
        warn("This will overwrite any existing file(s) with the same name!")
        os.remove(outfile_path)

    if highmem is True:
        dataset = pd.read_csv(dataset, sep=",")
        # parse everything by default
        # "的 " is used here as a filler to parse "\nFOO" strings (en) correctly
        if column == None:
            for col in dataset.columns:
                if dataset[col].dtype == "object":
                    dataset[col] = [
                        " ".join(i).replace("的 ", "\n") for i in [
                            [i for i in text.replace("\n", "的 ").split(" ")
                             if not i in stopwords_en]
                                for text in dataset[col]
                            ]
                        ]
        # target a specific column to parse
        else:
            dataset[column] = [
                " ".join(i).replace("的 ", "\n") for i in [
                    [i for i in text.replace("\n", "的 ").split(" ")
                     if not i in stopwords_en]
                        for text in dataset[column]
                    ]
                ]
        dataset.to_csv(outfile_path, index=False)

    else:
        # this hits all columns!
        with open(outfile_path, mode="a+") as outfile:
            with open(dataset) as infile:
                for line in infile:
                    outfile.write(" ".join(
                        [i for i in line.replace("\n", "的 ").split(" ")
                         if not i in stopwords_en]
                        ).replace("的 ", "\n"))
    return outfile_path

def dataset_to_disk(dataset: Dataset, outfile_dir: str, name: str):
    """Take a 🤗 dataset object, path as output and write files to disk

    Args:
        dataset (Dataset): A ``HuggingFace`` ``datasets.Dataset`` object
        outfile_dir (str): Write the dataset files to this path
        name (str): The name of the split, ie ``train``, ``test``,
            ``validation``. The file names will correspond to these.
            Validation set is optional.

    Returns:
        None:

        Nothing is returned, this writes files directly to ``outfile_dir``.

        This is normally called by :py:func:`split_datasets` but can be used
        directly if needed. Files are written directly to disk in multiple
        formats for use in downstream operations, e.g. model training.
    """
    if os.path.exists(outfile_dir):
        warn("".join(["Overwriting contents in directory!: ", outfile_dir]))
    dataset.to_csv("".join([outfile_dir, "/", name, ".csv"]))
    dataset.to_json("".join([outfile_dir, "/", name, ".json"]))
    dataset.to_parquet("".join([outfile_dir, "/", name, ".parquet"]))
    dataset.save_to_disk("".join([outfile_dir, "/", name]))

def split_datasets(dataset: DatasetDict, outfile_dir: str, train: float,
                   test: float=0, val: float=0, shuffle: bool=False):
    """Split data into training | testing | validation sets

    Args:
        dataset (DatasetDict): A ``HuggingFace`` ``DatasetDict`` object
        outfile_dir (str): Write the dataset files to this path
        train (float): Proportion of dataset for training
        test (float): Proportion of dataset for testing
        val (float): Proportion of dataset for validation
        shuffle (bool): Shuffle the dataset before splitting

    Returns:
        DatasetDict:

        Returns a ``datasets.DatasetDict`` object with corresponding
        ``train | test | valid`` splits. Writes files to ``outfile_dir``.

        Specifying the validation set is optional. However, note that train +
        test + validation proportions must sum to 1!
        This calls :py:func:`dataset_to_disk` to write files to disk.
        File names will match the corresponding split: ``train | test | valid``
    """
    assert train + test + val == 1, "Proportions of datasets must sum to 1!"
    train_split = 1 - train
    test_split = 1 - test / (test + val)
    val_split = 1 - val / (test + val)

    train = dataset.train_test_split(test_size=train_split, shuffle=shuffle)
    if val > 0:
        test_valid = train['test'].train_test_split(test_size=test_split, shuffle=shuffle)
        data = DatasetDict({
            'train': train['train'],
            'test': test_valid['test'],
            'valid': test_valid['train'],
            })
        print("Writing training set to disk...")
        dataset_to_disk(data["train"], outfile_dir, "train")
        print("Writing testing set to disk...")
        dataset_to_disk(data["test"], outfile_dir, "test")
        print("Writing validation set to disk...")
        dataset_to_disk(data["valid"], outfile_dir, "valid")
        return data
    else:
        data = DatasetDict({
            'train': train['train'],
            'test': train['test'],
            })
        print("Writing training set to disk...")
        dataset_to_disk(data["train"], outfile_dir, "train")
        print("Writing testing set to disk...")
        dataset_to_disk(data["test"], outfile_dir, "test")
        return data

def plot_hist(compare: list, outfile_path: str=None):
    """Plot histogram of alphas. Writes plot directly to disk. Also see
    :py:func:`plot_scatter`

    Args:
        compare (list[pd.DataFrame]): Paths to pandas dataframes with model info
        outfile_path (str): Write the plot to this path

    Returns:
        None:

        Smaller alpha is better [2, 4]. Computer Vision best models are ~2.
        If at least 1 layer has a score approaching 0, this indicates
        scale collapse.
        NLP models in the HuggingFace ``transformers`` library are
        deliberately overparameterised as they are intended as a base for
        fine tuning and are not a complete model. You will see values of
        [2, 6] before these are fine tuned, this is expected behaviour.

        If you want to compare your models against existing ones in HuggingFace
        as a quick comparison, you can download a model to disk, substituting
        out your model of interest as needed in the example below, then you can
        pass the path to the model as an argument to ``compare``::

            from transformers import DistilBertModel
            model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            model.save_pretrained("/path/to/distilbert")
    """
    def _plot_single(model_info, model_name: str=None, color: str=None):
        """Helper function to automate plotting of individual models."""
        model_info.alpha.plot.hist(bins=100, label=model_name, density=True,)# color=color)
        plt.axvline(model_info.alpha.mean(), linestyle='dashed',) # color=color,)

    for name, info in compare:
        _plot_single(info, name)

    plt.legend()
    plt.savefig(outfile_path, dpi=300)
    plt.close()

def plot_scatter(compare: list, outfile_path: str=None):
    """Plot scatterplot of alphas. Writes plot directly to disk. Also see
    :py:func:`plot_hist`

    Args:
        compare (list[pd.DataFrame]): Paths to pandas dataframes with model info
        outfile_path (str): Write the plot to this path

    Returns:
        None:

        Smaller alpha is better [2, 4]. Computer Vision best models are ~2.
        If at least 1 layer has a score approaching 0, this indicates
        scale collapse.
        NLP models in the HuggingFace ``transformers`` library are
        deliberately overparameterised as they are intended as a base for
        fine tuning and are not a complete model. You will see values of
        [2, 6] before these are fine tuned, this is expected behaviour.

        If you want to compare your models against existing ones in HuggingFace
        as a quick comparison, you can download a model to disk, substituting
        out your model of interest as needed in the example below, then you can
        pass the path to the model as an argument to ``compare``::

            from transformers import DistilBertModel
            model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            model.save_pretrained("/path/to/distilbert")
    """
    def _plot_single(model_info, model_name: str=None, color: str=None):
        """Helper function to automate plotting of individual models."""
        x = model_info.layer_id.to_numpy()
        y = model_info.alpha.to_numpy()
        plt.scatter(x, y, label=model_name)# color=color,)
        plt.axhline(np.mean(y), linestyle='dashed',)# color=color,)

    for name, info in compare:
        _plot_single(info, name)

    plt.legend()
    plt.savefig(outfile_path, dpi=300)
    plt.close()

def html_to_pdf(infile_path: str, outfile_path: str=None, options: dict=None):
    """Convert the output of transformers interpret to pdf and write to disk.

    Args:
        infile_path (str): path to transformers-interpret html output
        outfile_path (str): path to transformers-interpret pdf output
        options (dict): html to pdf conversion options

    Returns:
        None:

        Both pdfkit and wkhtmltopdf are required. Mainly used with `interpret`.
        Please refer to https://github.com/JazzCore/python-pdfkit::

            import pdfkit
            pdfkit.from_file("input.html", "output.pdf", options={...})
    """
    warning_message = "".join([
        "Both pdfkit and wkhtmltopdf are required. ", 
        "Please refer to https://github.com/JazzCore/python-pdfkit ",
        "for install instructions. No pdf output will be generated!"
    ])
    try:
        import pdfkit
    except FileNotFoundError:
        warn(warning_message)
        return        
    except ImportError:
        warn(warning_message)
        return
    except ModuleNotFoundError:
        warn(warning_message)
        return

    if options is None:
        options = {'dpi': 300, 'page-size': 'A6', 'orientation': 'landscape'}
    if outfile_path is None:
        outfile_path = ".".join([infile_path.replace(".html", ""), "pdf"])

    pdfkit.from_file(
        infile_path, 
        outfile_path, 
        options
        )