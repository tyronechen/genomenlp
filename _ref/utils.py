#!/usr/bin/python
# stores utility functions
import os
import re
import warnings
from joblib import parallel_backend, Parallel, delayed, dump, load
from random import shuffle
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import sklearn
import sklearn.pipeline
from sklearn import manifold, metrics, feature_selection as fs
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm

def bootstrap_seq(seq: str, block_size: int=2):
    """Take a string and shuffle it in blocks of N length"""
    chunks, chunk_size = len(seq), block_size
    seq = seq.replace(" ", "")
    seq = [ seq[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    shuffle(seq)
    return "".join(seq)

def build_kmers(sequence: str, ksize: int=3) -> str:
    """Generator that takes a fasta sequence and kmer size to return kmers"""
    for i in range(len(sequence) - ksize + 1):
        yield sequence[i:i + ksize], i, i + ksize

def select_features(train_vectors, train_lime, test_lime, vectorizer):
    """
    Perform feature selection given train vectors, data, vectorizer.
      this code was adapted from https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794?gi=1d88420af7c8
    """
    target = train_lime["target"]
    X_names = vectorizer.get_feature_names()
    feat = pd.DataFrame()
    for i in np.unique(target):
        chi2, p = fs.chi2(train_vectors, target==i)
        feat = feat.append(pd.DataFrame(
            {"feature": X_names, "score": 1-p, "y": i})
            )
        feat = feat.sort_values(["y","score"], ascending=[True,False])
        feat = feat[feat["score"] > feature_selection]
    X_names = feat["feature"].unique().tolist()

    # show a few of the selected features
    for i in np.unique(target):
       print("# {}:".format(i))
       print(" number of selected features:", len(feat[feat["y"]==i]))
       print(" top:", ",".join(feat[feat["y"]==i]["feature"].values[:10]))
    if algorithm == "cvec":
        vectorizer = CountVectorizer(lowercase=False, vocabulary=X_names)
    if algorithm == "tfidf":
        vectorizer = TfidfVectorizer(lowercase=False, vocabulary=X_names)
    vectorizer.fit(train_lime["data"])
    train_vectors = vectorizer.transform(train_lime["data"])
    test_vectors = vectorizer.transform(test_lime["data"])
    print("Matrix size (reduced features):", train_vectors.todense().shape)
    return train_vectors, test_vectors, train_lime, test_lime, vectorizer

def _write_output(data: dict, index: int, pipeline: sklearn.pipeline,
                  explainer: LimeTextExplainer, outfile_dir: str,
                  file_seqs_map: dict=None):
    """Catch output of explanation as html"""
    options = Options()
    options.add_argument('--headless')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outfile = [outfile_dir, "/", data["DESCR"][index], ".html"]
        outfile_path = "".join(outfile)
        exp = explainer.explain_instance(
            data["data"][index], pipeline.predict_proba, num_features=8
            )
        exp.save_to_file(outfile_path)
        # we can stop here if we just use kmers, the html will display correctly

        # ngram not very efficient since we parse html directly and double files
        if file_seqs_map is not None:
            outfile_real = [outfile_dir, "/", data["DESCR"][index], ".real.html"]
            outfile_real = "".join(outfile_real)
            with open(outfile_path, mode='r') as infile:
                with open(outfile_real, mode='w') as outfile:
                    for line in infile:
                        if re.match('^\s+exp.show', line):
                            for x, y in file_seqs_map[data["DESCR"][index]].items():
                                line = line.replace(x, y)
                            outfile.write(line)
                        else:
                            outfile.write(line)

                # we want to correct the highlights in the visualised output
                # html is generated from javascript code so we need to execute
                driver = webdriver.Firefox(options=options)
                outfile_absp = os.path.abspath(outfile_real)
                outfile_absp = "".join(["file://", outfile_absp])
                driver.get(outfile_absp)
                html = driver.execute_script(
                    "return document.getElementsByTagName('html')[0].innerHTML"
                    )
                driver.close()
                # now manually remove the wrong highlights
                soup = BeautifulSoup(html, 'html.parser')
                high_weights = soup.find_all("text", {"font-size":14})
                high_weights = [str(x.text) for x in high_weights]
                for i in soup.div.findAll("span"):
                    i.unwrap()
                # and replace it with the correct highlights (all equal opacity)
                text = re.search(r'</h3>(.*)</div></div>', str(soup.div)).group(1)
                original = re.search(r'(.*</h3>)', str(soup.div)).group(1)
                leading = "<span style=\"white-space: pre-wrap;\">"
                colour = "<span style=\"background-color: rgb(255, 127, 14);\">"
                trailing = "</span></div></div>"
                for w in high_weights:
                    highlight = "".join([leading, colour, w, trailing])
                    text = text.replace(w, highlight)
                html_out = "".join([original, text])
                with open(outfile_path, mode='w') as outfile:
                    outfile.write(html_out)
                os.remove(outfile_real)

def map_synthetic_real(data: pd.DataFrame, ksize: int, ngram: int, threads: int,
                       hide_progressbar: bool=False):
    """Generate kmers, if ngrams generate a map of real:synthetic seqs"""
    # build k-mers
    kmers = [build_kmers(x, ksize) for x in data["data"].tolist()]
    count = len(kmers)
    assert count == len(data), "Number of seqs must match kmer collections"

    if ngram == 0:
        for x in range(count):
            data["data"][x] = " ".join([y[0] for y in kmers[x]])

    # ngram version written differently to improve efficiency
    if ngram > 0:
        file_seqs_map = dict()
        if hide_progressbar is False:
            for x in tqdm(range(count), desc="Creating n-gram mapping"):
                y = [_[0] for _ in kmers[x]]
                ngrams = build_kmers(y, ngram)
                ngrams = ["".join(z) for z in [_[0] for _ in ngrams]]
                synthetic_real_map = dict(Parallel(n_jobs=threads)(
                    delayed(_map_synthetic_real)(_, ngram) for _ in ngrams
                    ))
                data["data"][x] = " ".join(ngrams)
                file_seqs_map[data["DESCR"][x]] = synthetic_real_map
        else:
            print("Creating n-gram mapping...")
            for x in range(count):
                y = [_[0] for _ in kmers[x]]
                ngrams = build_kmers(y, ngram)
                ngrams = ["".join(z) for z in [_[0] for _ in ngrams]]
                synthetic_real_map = dict(Parallel(n_jobs=threads)(
                    delayed(_map_synthetic_real)(_, ngram) for _ in ngrams
                    ))
                data["data"][x] = " ".join(ngrams)
                file_seqs_map[data["DESCR"][x]] = synthetic_real_map
    else:
        file_seqs_map = None
    return data, file_seqs_map

def _map_synthetic_real(n: str, ngram: int):
    """Map the synthetic sequences back to the original"""
    stride = len(n) // ngram
    return n, n[:stride-1] + n[stride-1::stride]

def plot_roc(pipeline: sklearn.pipeline, test_data: pd.DataFrame, outdir: str):
    """Plot ROC curves"""
    # overall accuracy
    acc = pipeline.score(test_data["data"], test_data["target"])

    # get roc/auc info
    Y_score = pipeline.predict_proba(test_data["data"])[:,1]
    fpr = dict()
    tpr = dict()
    fpr, tpr, _ = roc_curve(test_data["target"], Y_score)

    roc_auc = dict()
    roc_auc = auc(fpr, tpr)

    # make the plot
    plt.figure(figsize=(10,10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.plot(fpr, tpr, label='AUC = {0}'.format(roc_auc))
    plt.legend(loc="lower right", shadow=True, fancybox =True)
    plt.savefig("/".join([outdir, "roc.pdf"]), format="pdf")

def show_embedding_word(model, word: str, savefig: str="./3d_embedding.pdf"):
    fig = plt.figure()## word embedding
    tot_words = [word] + [tupla[0] for tupla in model.wv.most_similar(word, topn=20)]
    X = model.wv[tot_words]## pca to reduce dimensionality from 300 to 3
    pca = manifold.TSNE(perplexity=40, n_components=3, init='pca')
    X = pca.fit_transform(X)## create dtf
    dtf_ = pd.DataFrame(X, index=tot_words, columns=["x","y","z"])
    dtf_["input"] = 0
    dtf_["input"].iloc[0:1] = 1## plot 3d
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dtf_[dtf_["input"]==0]['x'],
               dtf_[dtf_["input"]==0]['y'],
               dtf_[dtf_["input"]==0]['z'], c="black")
    ax.scatter(dtf_[dtf_["input"]==1]['x'],
               dtf_[dtf_["input"]==1]['y'],
               dtf_[dtf_["input"]==1]['z'], c="red")
    ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[],
           yticklabels=[], zticklabels=[])
    for label, row in dtf_[["x","y","z"]].iterrows():
        x, y, z = row
        ax.text(x, y, z, s=label)
    plt.title(word)
    plt.savefig(savefig, dpi=300)
    plt.close()

def show_acc_loss(trained, savefig: str="./acc_loss.pdf"):
    ## plot loss and accuracy
    metrics = [k for k in trained.history.keys()
               if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(trained.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(trained.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()

    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(trained.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
         ax22.plot(trained.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    ax22.legend()
    plt.savefig(savefig, dpi=300)
    plt.close()

def show_summary_stats(y_test, predicted, predicted_prob, savefig="./summary_stats.pdf"):
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values
    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob[:,1])
    print("Accuracy:", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))

    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
           yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test_array[:,i], predicted_prob[:,i]
            )
        ax[0].plot(fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(
            classes[i], metrics.auc(fpr, tpr))
            )
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05],
              xlabel='False Positive Rate',
              ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)

    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test_array[:,i], predicted_prob[:,i]
            )
        ax[1].plot(
            recall, precision, lw=3, label='{0} (area={1:0.2f})'.format(
                classes[i], metrics.auc(recall, precision))
                )
    ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall',
              ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.savefig(savefig, dpi=300)
    plt.close()

def show_sample_vector(train: pd.DataFrame, X_train, vocab: dict, i: int=0):
    """Show example word:vector mapping for debugging."""
    if type(train) == pd.DataFrame:
        original = train["data"].iloc[i]
    if type(train) == dict:
        original = train["data"]
    print("original:", original)
    len_tokens = len(X_train[i])
    print("vectored:", X_train[i], X_train)
    print(original[i], "==", vocab[original[i]])
    print("vocabulary size:", len(vocab.items()))

def _tokenise_seq(seq: str, ksize: int, hide_progressbar: bool=False):
    """Tokenise the seq into k-length blocks."""
    if hide_progressbar is True:
        for j in tqdm(range(0, len(seq))):
            x = seq[j:j + ksize]
            if len(x) == ksize:
                yield x
    else:
        for j in range(0, len(seq)):
            x = seq[j:j + ksize]
            if len(x) == ksize:
                yield x

def _tokenise_seqs(fadb, index: int, ksize: int, hide_progressbar: bool=False):
    """Tokenise the seq collection into k-length blocks."""
    seq = fadb.loadRecordByIndex(index)['sequence']
    seq = list(_tokenise_seq(seq, ksize, hide_progressbar))
    yield seq
