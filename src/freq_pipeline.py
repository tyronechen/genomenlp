import os
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
from yellowbrick.text import FreqDistVisualizer
from utils import parse_sp_tokenised

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

def _run_search(model, param, x_train, y_train, x_test, y_pred, feature,
                n_top_features, n_jobs):
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

def train_model(model, param, x_train, y_train, x_test):
   clf=model(param)
   # fit the training data into the model
   clf.fit(x_train, y_train)
   y_pred=clf.predict(x_test)
   y_probas=clf.predict_proba(x_test)
   return clf, y_pred, y_probas

def main():
    n_gram_from= 1
    n_gram_to= 1
    split_train = 0.90
    split_test = 0.05
    split_val = 0.05
    kfolds = 8
    sweep_count = 128
    model = RandomForestClassifier
    freq_method = "tfidf"
    scoring = "f1"
    n_jobs = 6
    # ["accuracy", "f1", "precision", "recall"]
    metric_opt = "f1"
    param = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30, 40, 50],
        'min_samples_split': [2, 3, 5, 7, 11],
        'min_samples_leaf': [2, 3, 5, 7, 11],
        'bootstrap': [True, False],
        }
    infile_path = [
        '../results/tmp/train.csv',
        '../results/tmp/test.csv',
        '../results/tmp/valid.csv'
        ]
    tokeniser_path = "../results/tmp/yeast.json"
    output_dir = "./"

    # load data and parse tokens using intended strategy (SP or k-merisation)
    if tokeniser_path != None:
        tokens = pd.concat(
            [parse_sp_tokenised(x, tokeniser_path) for x in infile_path]
            )
        tokens.reset_index(drop=True, inplace=True)
        dna = tokens[['input_str', 'labels']]
        corpus = dna['input_str'].apply(lambda x: " ".join(x)).tolist()

    # choose frequency vectoriser method, weighted (tfidf) or nonweighted (cvec)
    if freq_method == "tfidf":
        vectoriser = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(n_gram_from, n_gram_to),
            smooth_idf=False,
            lowercase=False,
            )
        vectorised = vectoriser.fit_transform(corpus)
    if freq_method == "cvec":
        vectoriser = CountVectorizer(
            max_features=max_features,
            ngram_range=(n_gram_from, n_gram_to)
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
        if sweep == "grid":
            param_instances = [i for i in ParameterGrid(param)][:sweep_count]
            metrics_sweep = [
                _run_search(model, i, x_train, y_train, x_test, y_pred, feature,
                n_top_features, n_jobs) for i in tqdm(param_instances,
                                                      desc="Grid Search:")
                ]
        if sweep == "random":
            metrics_sweep = [
                _run_search(model, i, x_train, y_train, x_test, y_pred, feature,
                n_top_features, n_jobs)
                for i in tqdm(ParameterSampler(param, n_iter=sweep_count),
                                                      desc="Random Search:")
                ]
        if sweep == "bayes":
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

    dump(clf, "/".join([output_dir, "model.joblib"]))

    # perform cross validation on best model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cval_scores = cross_val_score(
            clf, x_train, y_train, cv=8, scoring="roc_auc", n_jobs=n_jobs
            )
    cval_scores = pd.DataFrame(cval_scores)
    cval_scores.columns = ["roc_auc_scores"]
    cval_scores.to_csv("".join([output_dir, "/cval_auc.tsv",]), index=False)

if __name__ == "__main__":
    main()
