#!/usr/bin/python
# take custom formatted bedfile, break into kmers for classification with lime
import argparse
from math import floor
import os
import re
import warnings
from warnings import warn
from joblib import parallel_backend, Parallel, delayed, dump, load
import json
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_text import LimeTextExplainer
import pandas as pd
import seaborn as sns
import sklearn.ensemble
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import sklearn.metrics
from utils import bootstrap_seq, build_kmers, plot_roc, map_synthetic_real, select_features, _map_synthetic_real, _write_output

def main():
    parser = argparse.ArgumentParser(
        description='Take custom formatted bedfile, classify kmers with lime.'
    )
    parser.add_argument('-i', '--infile_path', type=str, default=None,
                        help='path to bed-like file with data')
    parser.add_argument('-b', '--block_size', type=int, default=2,
                        help='size of block to permute null seqs (DEFAULT: 2)')
    parser.add_argument('-c', '--control_set', type=str, default=None,
                        help='use an existing bed-like file as a null set')
    parser.add_argument('--train_data', type=str, default=None,
                        help='use an existing bed-like file as a training set')
    parser.add_argument('--test_data', type=str, default=None,
                        help='use an existing bed-like file as a testing set')
    parser.add_argument('-s', '--sample_classes', type=str, nargs="+",
                        default=["control", "test"],
                        help='sample classes (DEFAULT: ["control", "test"])')
    parser.add_argument('-w', '--window_size', type=int, default=5,
                        help='size of sliding window to take (DEFAULT: 5)')
    parser.add_argument('-a', '--algorithm', type=str, default="tfidf",
                        help='choose algorithm [tfidf | cvec | w2v] (DEFAULT: tfidf)')
    parser.add_argument('-m', '--model_load', type=str, default=None,
                        help='load model from this path (DEFAULT: None)')
    parser.add_argument('-v', '--vectorizer_load', type=str, default=None,
                        help='load vectorizer from this path (DEFAULT: None)')
    parser.add_argument('-n', '--ngram_count', type=int, default=0,
                        help='number of kmers to include in ngram (DEFAULT: 0)')
    parser.add_argument('-fs', '--feature_selection', type=float, default=None,
                        help='chi square test for feature:target independence (DEFAULT: None)')
    parser.add_argument('-t', '--threads', type=int, default=2,
                        help='number of cpus to use (DEFAULT: 2)')
    parser.add_argument('-f', '--force_if_lowscore', action="store_true",
                        help='force continue even if acc low (DEFAULT: False)')
    parser.add_argument('-d', '--display_count', type=int, default=0,
                        help='number of test data to show (DEFAULT: 0 [ALL])')
    parser.add_argument('-o', '--outfile_dir', type=str, default=None,
                        help='write html output to this dir (DEFAULT: None)')
    parser.add_argument('-p', '--hide_progressbar', action="store_true",
                        help='hide the progress bar (DEFAULT: False)')

    args = parser.parse_args()
    infile_path = args.infile_path
    threads = args.threads
    outdir = args.outfile_dir
    ksize = args.window_size
    load_model = args.model_load
    load_vectorizer = args.vectorizer_load
    train_data = args.train_data
    test_data = args.test_data
    ngram = args.ngram_count
    control = args.control_set
    sample_classes = args.sample_classes
    display = args.display_count
    hide = args.hide_progressbar
    algorithm = args.algorithm
    feature_selection = args.feature_selection

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if load_model and train_data or \
        load_model and infile_path or \
        train_data and infile_path:
        message = " ".join([
            "model_load, train_data, infile_path are mutually exclusive!"
            "Priority is given to model_load > train_data > infile_path,"
            "all arguments lower in the sequence will be overriden!"
            ])
        warn(message)

    # this bed-like file is a custom format with the header removed from:
    # http://regulondb.ccg.unam.mx/menu/download/datasets/files/PromoterSet.txt
    if train_data:
        data = pd.read_csv(train_data, sep="\t", header=None)
    elif infile_path:
        data = pd.read_csv(infile_path, sep="\t", header=None)

    data = data[[1, 6]]
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    data[6] = data[6].str.lower()
    data[2] = 1
    data.columns = ["DESCR", "data", "target"]

    if load_model is None or load_vectorizer is None:
        print("No model provided, training new model.")
        # to create a null distribution we shuffle the seqs in blocks of length n
        if control is None:
            null = pd.DataFrame()
            null["DESCR"] = "NULL_" + data["DESCR"]
            null.dropna(inplace=True)
            null.reset_index(inplace=True, drop=True)
            null["data"] = [
                bootstrap_seq(x, args.block_size) for x in data["data"].tolist()
                ]
            null["target"] = 0
        else:
            null = pd.read_csv(args.infile_path, sep="\t", header=None)
            null = null[[1, 6]]
            null.dropna(inplace=True)
            null.reset_index(inplace=True, drop=True)
            null[6] = null[6].str.lower()
            null[2] = 0
            null.columns = ["DESCR", "data", "target"]
        data = pd.concat([data, null]).reset_index().drop("index", axis=1)
    else:
        print("Model provided, loading from model (overrides other arguments!)")
        rf = load(load_model)
        vectorizer = load(load_vectorizer)

    # multithreading actually makes this process longer, disable
    data, file_seqs_map = map_synthetic_real(
        data=data, ksize=ksize, ngram=ngram, threads=1, hide_progressbar=hide
        )

    categories = sample_classes
    class_names = sample_classes

    if load_model is None or load_vectorizer is None:
        if train_data is None or test_data is None:
            # divide into training and testing sets (hardcoded to 2 class only)
            class_size = len(data) // 2
            test_size = floor(1 / 10 * class_size)

            test_seqs = data[0:test_size]
            test_null = data[class_size:class_size+test_size]

            train_seqs = data[test_size:class_size]
            train_null = data[class_size+test_size:]

            train_data = pd.concat([train_seqs, train_null])
            test_data = pd.concat([test_seqs, test_null])
        else:
            test_data = pd.read_csv(test_data, sep="\t", header=None)
            test_data = test_data[[1, 6]]
            test_data.dropna(inplace=True)
            test_data.reset_index(inplace=True, drop=True)
            test_data[6] = test_data[6].str.lower()
            test_data[2] = 1
            test_data.columns = ["DESCR", "data", "target"]
            test_data, file_seqs_map = map_synthetic_real(
                data=test_data, ksize=ksize, ngram=ngram, threads=threads
                )

        # convert data into a format compatible with the lime pipeline
        train_lime = dict()
        train_data = data
        for i in train_data.columns:
            train_lime[i] = train_data[i].tolist()
        # train_lime["target_names"] = categories

        test_lime = dict()
        for i in test_data.columns:
            test_lime[i] = test_data[i].tolist()
        # test_lime["target_names"] = categories

        # vectorise data and pass through random forest
        #   following steps are adapted from the lime pipeline tutorial here:
        #   https://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html
        if algorithm == "cvec":
            vectorizer = CountVectorizer(lowercase=False)
        if algorithm == "tfidf":
            vectorizer = TfidfVectorizer(lowercase=False)
        train_vectors = vectorizer.fit_transform(train_lime["data"])
        test_vectors = vectorizer.transform(test_lime["data"])
    else:
        test_seqs = data
        test_data = test_seqs
        test_lime = dict()
        for i in test_data.columns:
            test_lime[i] = test_data[i].tolist()
        # test_lime["target_names"] = ["control", "test"]
        test_vectors = vectorizer.transform(test_lime["data"])

    # print(pd.DataFrame.from_dict(train_lime)[:10]["data"])
    # print(pd.DataFrame.from_dict(train_lime)[5306:5316]["data"])

    # perform feature selection to reduce vocabulary size
    print("Matrix size (all features):", train_vectors.todense().shape)
    if feature_selection is not None:
        train_vectors, test_vectors, train_lime, test_lime, vectorizer = \
            select_features(train_vectors, train_lime, test_lime, vectorizer)
    else:
        print("Skipping feature selection (keep all)")

    print(test_lime["data"][0])
    print(len(test_lime["data"][0]))

    if load_model is None or load_vectorizer is None:
        # TODO: enable multiple different nn architectures
        with parallel_backend('threading', n_jobs=threads):
            rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
            rf.fit(train_vectors, train_data["target"])
            # y_score = y_score.decision_function(test_lime["data"])
        if outdir is not None:
            dump(rf, "/".join([outdir, "model.joblib"]))
            dump(vectorizer, "/".join([outdir, "vectorizer.joblib"]))
        else:
            dump(rf, "model.joblib")
            dump(vectorizer, "vectorizer.joblib")

    pred = rf.predict(test_vectors)
    pred_proba = rf.predict_proba(test_vectors)
    f1 = sklearn.metrics.f1_score(test_data["target"], pred, average='binary')
    # report = sklearn.metrics.classification_report(test_data["target"], pred)
    # print("Prediction accuracy on test data:")
    # print(report)
    report = sklearn.metrics.classification_report(
        test_data["target"], pred, output_dict=True
        )
    if outdir:
        outfile_path = "/".join([outdir, "classification_report.json"])
        with open(outfile_path, mode="w") as outfile:
            outfile.write(json.dumps(report))
        imp_path = "/".join([outdir, "feature_importances.tsv"])
        imp = pd.Series(rf.feature_importances_)
        imp = pd.DataFrame(imp.sort_values(ascending=False))
        print("Important features:", len(imp))
        imp.to_csv(imp_path, sep="\t")
        cmat = confusion_matrix(test_data["target"], pred)
        print("Confusion matrix:", "\n", cmat)
        np.savetxt("/".join([outdir, "confusion.tsv"]), cmat, delimiter='\t')
        cv = np.mean(cross_val_score(
            rf, train_vectors, train_data["target"], cv=10,
            scoring='recall_macro', n_jobs=threads
            ))
        print("Cross-validation", "\n", cv)
        cv_path = "/".join([outdir, "cv.txt"])
        with open(cv_path, mode="w") as outfile:
            outfile.write(str(cv))

        # adapted from https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794?gi=1d88420af7c8
        y = test_lime["target"]
        classes = np.unique(y)
        y_tarray = pd.get_dummies(y, drop_first=False).values
        accuracy = metrics.accuracy_score(y, pred)
        auc = metrics.roc_auc_score(y, pred_proba[:, 1])#, multi_class="ovr")
        print("Accuracy:", round(accuracy, 2))
        print("Auc:", round(auc, 2))
        print("Detail:")
        print(metrics.classification_report(y, pred))

        cm = metrics.confusion_matrix(y, pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                    cbar=False)
        ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
               yticklabels=classes, title="Confusion matrix")
        plt.yticks(rotation=0)
        confusion_path = "/".join([outdir, "confusion.pdf"])
        plt.savefig(confusion_path, dpi=300)
        plt.close()

        fig, ax = plt.subplots(nrows=1, ncols=2)
        for i in range(len(categories)):
            fpr, tpr, thr = metrics.roc_curve(y_tarray[:,i], pred_proba[:,i])
            ax[0].plot(fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(
                classes[i], metrics.auc(fpr, tpr)
                ))
            ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
            ax[0].set(
                xlim=[-0.05,1.0], ylim=[0.0,1.05], xlabel='False Positive Rate',
                ylabel="True Positive Rate (Recall)",
                title="Receiver operating characteristic"
                )
            ax[0].legend(loc="lower right")
            ax[0].grid(True)

        for i in range(len(classes)):
            precision, recall, thr = metrics.precision_recall_curve(
                y_tarray[:,i], pred_proba[:,i]
                )
            ax[1].plot(recall,precision,lw=3,label='{0} (area={1:0.2f})'.format(
                classes[i], metrics.auc(recall, precision)
                ))
            ax[1].set(
                xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall',
                ylabel="Precision", title="Precision-Recall curve")
            ax[1].legend(loc="best")
            ax[1].grid(True)
        metrics_path = "/".join([outdir, "classification_report.pdf"])
        plt.savefig(metrics_path, dpi=300)
        plt.close()

        vocab_path = "/".join([outdir, "vocabulary_size.pdf"])
        box = plt.figure()
        pd.DataFrame(pd.Series(vectorizer.vocabulary_)).boxplot()
        box = plt.savefig(vocab_path, dpi=300)
        plt.close()

    if f1 < 70.0:
        if args.force_if_lowscore is True:
            warn("Prediction accuracy is low!")
        else:
            raise RuntimeError(
                "Prediction accuracy is low! Force continue with -f"
                )

    # lime provides interpretable html output
    if outdir is not None:
        warn("Writing output can take a long time (1 for each test sample!)")
        c = make_pipeline(vectorizer, rf)
        explainer = LimeTextExplainer(class_names=class_names)
        desc = "".join(["Writing data to: ", outdir])
        print(desc)
        plot_roc(c, test_data, outdir)

        if display == 0:
            display = len(test_lime["data"])

        Parallel(n_jobs=threads)(
            delayed(_write_output)(
                test_lime, _, c, explainer, outdir, file_seqs_map
                )
            for _ in range(display)
            )

if __name__ == '__main__':
    main()
