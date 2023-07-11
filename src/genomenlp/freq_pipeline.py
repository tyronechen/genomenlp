#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

import argparse
import json
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, plot_confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, ParameterGrid, ParameterSampler, cross_val_score, StratifiedKFold, cross_validate
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, AutoModel
from utils import parse_sp_tokenised, get_feature_importance_mdi, get_feature_importance_per, _cite_me
from xgboost import XGBClassifier
from yellowbrick.text import FreqDistVisualizer

def _compute_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    precision_0 = precision_score(y_test, y_pred, pos_label=0)
    precision_1 = precision_score(y_test, y_pred, pos_label=1)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": accuracy,
        "f1_0": f1_0,
        "f1_1": f1_1,
        "precision_0": precision_0,
        "precision_1": precision_1,
        "recall_0": recall_0,
        "recall_1": recall_1,
        "roc_auc": roc_auc,
        "conf_mat": conf_mat,
    }

def _run_search(model, param, x_train, y_train, x_test, y_test, feature,
                n_top_features=100, n_jobs=-1):
    "Helper function to run search, not to be used directly"
    clf = model(
        n_estimators=param["n_estimators"],
        min_samples_split=param["min_samples_split"],
        min_samples_leaf=param["min_samples_leaf"],
        max_features=param["max_features"],
        max_depth=param["max_depth"],
        bootstrap=param["bootstrap"],
        n_jobs=n_jobs
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_probas = clf.predict_proba(x_test)
    metrics = _compute_metrics(y_test, y_pred)
    # compute_feature_importances(clf, feature, n_top_features)
    return {"param": param, "metrics": metrics,
            "y_pred": y_pred, "y_probas": y_probas}

def compute_feature_importances(model, feature, n_top_features, outfile_path=None):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.barh(
        np.array(feature)[indices][:n_top_features],
        importances[indices][:n_top_features]
        )
    plt.xlabel("RF feature Importance")
    if outfile_path != None:
        plt.savefig(outfile_path)
    plt.clf()

def token_freq_plot(feature, X):
    visualizer = FreqDistVisualizer(features=feature, orient='v')
    visualizer.fit(X)
    visualizer.show()

def main():
    parser = argparse.ArgumentParser(
         description='Take HuggingFace dataset and perform parameter sweeping.'
        )
    parser.add_argument('--infile_path', type=str, nargs="+", default=None,
                        help='path to [ csv | csv.gz | json | parquet ] file')
    parser.add_argument('--format', type=str, default="csv",
                        help='specify input file type [ csv | json | parquet ]')
    parser.add_argument('--embeddings', type=str, default=None,
                        help='path to embeddings model file')
    parser.add_argument('--chunk_size', type=int, default=9999999999999999,
                        help='iterate over input file for these many rows')
    parser.add_argument('-t', '--tokeniser_path', type=str, default=None,
                        help='path to tokeniser.json file to load data from')
    parser.add_argument('-f', '--freq_method', type=str, default="tfidf",
                        help='choose dist [ cvec | tfidf ] (DEFAULT: tfidf)')
    parser.add_argument('--column_names', type=str, default=["idx",
                        "feature", "labels", "input_ids", "token_type_ids",
                        "attention_mask", "input_str"],
                        help='column name for sp tokenised data \
                        (DEFAULT: input_str)')
    parser.add_argument('--column_name', type=str, default="input_str",
                        help='column name for extracting embeddings \
                        (DEFAULT: input_str)')
    parser.add_argument('-m', '--model', type=str, default="rf",
                        help='choose model [ rf | xg ] (DEFAULT: rf)')
    parser.add_argument('-e', '--model_features', type=int, default=None,
                        help='number of features in data to use (DEFAULT: ALL)')
    parser.add_argument('-k', '--kfolds', type=int, default=8,
                        help='number of cross validation folds (DEFAULT: 8)')
    parser.add_argument('--show_features', type=int, default=50,
                        help='number of shown feature importance (DEFAULT: 50)')
    parser.add_argument('--ngram_from', type=int, default=1,
                        help='ngram slice starting index (DEFAULT: 1)')
    parser.add_argument('--ngram_to', type=int, default=1,
                        help='ngram slice ending index (DEFAULT: 1)')
    parser.add_argument('--split_train', type=float, default=0.90,
                        help='proportion of training data (DEFAULT: 0.90)')
    parser.add_argument('--split_test', type=float, default=0.05,
                        help='proportion of testing data (DEFAULT: 0.05)')
    parser.add_argument('--split_val', type=float, default=0.05,
                        help='proportion of validation data (DEFAULT: 0.05)')
    parser.add_argument('-o', '--output_dir', type=str, default="./results_out",
                        help='specify path for output (DEFAULT: ./results_out)')
    parser.add_argument('-s', '--vocab_size', type=int, default=32000,
                        help='vocabulary size for model configuration')
    parser.add_argument('--special_tokens', type=str, nargs="+",
                        default=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                        help='assign special tokens, eg space and pad tokens \
                        (DEFAULT: ["<s>", "</s>", "<unk>", "<pad>", "<mask>"])')
    parser.add_argument('-w', '--hyperparameter_sweep', type=str, default=None,
                        help='run a hyperparameter sweep with config from file')
    parser.add_argument('--sweep_method', type=str, default="random",
                        help='specify sweep search strategy \
                        [ bayes | grid | random ] (DEFAULT: random)')
    parser.add_argument('-n', '--sweep_count', type=int, default=8,
                        help='run n hyperparameter sweeps (DEFAULT: 8)')
    parser.add_argument('-c', '--metric_opt', type=str, default="f1",
                        help='score to maximise [ accuracy | f1 | precision | \
                        recall ] (DEFAULT: f1)')
    parser.add_argument('-j', '--njobs', type=int, default=-1,
                        help='run on n threads (DEFAULT: -1)')
    parser.add_argument('-d', '--pre_dispatch', default="0.5*n_jobs",
                        help='specify dispatched jobs (DEFAULT: 0.5*n_jobs)')
    args = parser.parse_args()

    infile_path = args.infile_path
    chunk_size = args.chunk_size
    column_name = args.column_name
    column_names = args.column_names
    format = args.format
    n_gram_from = args.ngram_from
    n_gram_to = args.ngram_to
    split_train = args.split_train
    split_test = args.split_test
    split_val = args.split_val
    kfolds = args.kfolds
    show_features = args.show_features
    model_features = args.model_features
    output_dir = args.output_dir
    vocab_size = args.vocab_size
    special_tokens = args.special_tokens
    param = args.hyperparameter_sweep
    sweep_method = args.sweep_method
    sweep_count = args.sweep_count
    metric_opt = args.metric_opt
    model = args.model
    freq_method = args.freq_method
    n_jobs = args.njobs
    pre_dispatch = args.pre_dispatch
    tokeniser_path = args.tokeniser_path

    print("\n\nARGUMENTS:\n", args, "\n\n")

    if infile_path == None:
        raise OSError("Require at least one input file path")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if model == "rf":
        model = RandomForestClassifier
        model_type = "rf"
    if model == "xg":
        model = XGBClassifier
        model_type = "xg"

    if param != None:
        with open(param, mode="r") as infile:
            param = json.load(infile)
    else:
        param = {
            'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'max_features': ["sqrt", "log2", None],
            'max_depth': [2, 3, 5, 7, 11],
            'min_samples_split': [2, 3, 5, 7, 11],
            'min_samples_leaf': [2, 3, 5, 7, 11],
            'bootstrap': [True, False],
            }

    # load data and parse tokens using intended strategy (SP or k-merisation)
    if tokeniser_path != None:
        output_files = list()
        output_data = list()
        for i in infile_path:
            output_files.append("/".join([output_dir, os.path.basename(i)]))
            output_data.append(parse_sp_tokenised(
                infile_path=i,
                outfile_path="/".join([output_dir, os.path.basename(i)]),
                tokeniser_path=tokeniser_path,
                special_tokens=special_tokens,
                chunksize=chunk_size,
                columns=column_names,
            ))
        tokens = pd.concat([pd.read_csv(i, index_col=0) for i in output_files])
        tokens.reset_index(drop=True, inplace=True)
        dna = tokens[['input_str', 'labels']]
        corpus = tokens['input_str'].apply(
            lambda x: x[1:-1].replace("\'", "")#.split()
            )
    else:
        pass

    # choose frequency vectoriser method, weighted (tfidf) or nonweighted (cvec)
    if freq_method == "tfidf":
        vectoriser = TfidfVectorizer(
            max_features=model_features,
            ngram_range=(n_gram_from, n_gram_to),
            smooth_idf=True,
            lowercase=False,
            )
        vectorised = vectoriser.fit_transform(corpus)
    if freq_method == "cvec":
        vectoriser = CountVectorizer(
            max_features=model_features,
            ngram_range=(n_gram_from, n_gram_to),
            lowercase=False,
            )
        vectorised = vectoriser.fit_transform(corpus)
    features = vectoriser.get_feature_names()
    vectorised = vectorised.toarray()
    vectorised = np.nan_to_num(vectorised)
    labels = np.array(dna["labels"])

    # assign training and testing splits, validation optional
    if split_val is None:
        split_val = 0
    assert split_train + split_test + split_val == 1, \
        "Proportions of datasets must sum to 1!"

    train_size = 1 - split_train
    test_size = 1 - split_test / (split_test + split_val)
    val_size = 1 - split_val / (split_test + split_val)

    # NOTE: train_size assigned to test_size is not a mistake
    x_train, x_test, y_train, y_test = train_test_split(
        vectorised, labels, test_size=train_size, shuffle=True,
        )
    if split_val > 0:
        x_val, x_test, y_val, y_test = train_test_split(
            x_test, y_test, test_size=val_size, shuffle=True
        )

    # print out stats for debugging
    print("Total data items:", vectorised.shape)
    print("Total data labels", labels.shape)
    print("Training data:",x_train.shape)
    print("Training data labels:",y_train.shape)
    print("Test data:",x_test.shape)
    print("Test data labels:",y_test.shape)
    print("Validation data:",x_val.shape)
    print("Validation data labels:",y_val.shape)

    # perform the hyperparameter sweeps using the specified algorithm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sweep_method == "grid":
            param_instances = [i for i in ParameterGrid(param)][:sweep_count]
            metrics_sweep = [
                _run_search(model, i, x_train, y_train, x_test, y_test, features,
                n_top_features=100, n_jobs=n_jobs) for i in tqdm(param_instances,
                                                      desc="Grid Search:")
                ]
        if sweep_method == "random":
            metrics_sweep = [
                _run_search(model, i, x_train, y_train, x_test, y_test, features,
                n_top_features=100, n_jobs=n_jobs)
                for i in tqdm(ParameterSampler(param, n_iter=sweep_count),
                                                      desc="Random Search:")
                ]
        if sweep_method == "bayes":
            pass

    # compile metrics from the corresponding sweeps for selecting best params
    metrics_all = pd.DataFrame([pd.Series(i["metrics"]) for i in metrics_sweep])
    params_all = pd.concat([pd.DataFrame(
        pd.Series(i["param"])).T for i in metrics_sweep]
        ).reset_index().drop("index", axis=1)
    metrics_params = pd.concat([metrics_all, params_all], axis=1)

    # pick the best scoring instance and get the corresponding parameters
    # NOTE: for metrics calculated on each class, the score is averaged for sort
    if metric_opt == "f1" or metric_opt == "precision" or metric_opt == "recall":
        metric_opts = ["_".join([metric_opt, str(i)]) for i in [0, 1]]
        metrics_params[metric_opt] = metrics_params[metric_opts].apply("mean", axis=1)
        best_params = metrics_params.sort_values(metric_opt, ascending=False).iloc[0]

    if metric_opt == "accuracy" or metric_opt == "roc_auc":
        best_params = metrics_params.sort_values(metric_opt, ascending=False).iloc[0]

    # train model on best parameters
    clf = model(
        n_estimators=best_params["n_estimators"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        max_depth=best_params["max_depth"],
        bootstrap=best_params["bootstrap"],
        n_jobs=n_jobs
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_probas = clf.predict_proba(x_test)
    metrics = _compute_metrics(y_test, y_pred)
    # compute_feature_importances(clf, feature, n_top_features)

    # save model
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if model_type == "rf":
        dump(clf, "/".join([output_dir, "model.joblib"]))
    if model_type == "xg":
        clf.save_model("/".join([output_dir, "model.json"]))

    mdi_scores = get_feature_importance_mdi(
        clf, np.array(features), model_type, show_features, output_dir,
        )
    per_scores = get_feature_importance_per(
        clf, x_test, y_test, np.array(features), model_type, show_features,
        output_dir, n_repeats=10, n_jobs=n_jobs
        )

    # perform cross validation on best model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cval_scores = cross_val_score(
            clf, x_train, y_train, cv=kfolds, scoring="roc_auc", n_jobs=n_jobs,
            pre_dispatch=pre_dispatch
            )
    cval_scores = pd.DataFrame(cval_scores)
    cval_scores.columns = ["roc_auc_scores"]
    cval_scores.to_csv("".join([output_dir, "/cval_auc.tsv",]), index=False)

if __name__ == "__main__":
    main()
    _cite_me()