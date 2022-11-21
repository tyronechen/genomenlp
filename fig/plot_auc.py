#!/usr/bin/python
# generate the roc auc scores in the paper
# data was obtained from 8 cross validation runs of the best model obtained by
#   hyperparameter sweeping
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_auc_scores(data: pd.DataFrame):
    vio_path = "auc_violin_plot.pdf"
    box_path = "auc_boxplot.pdf"
    fig = sns.violinplot(
        data=data,
        # x="group_name",
        # y=i,
        inner="box",
        # cut=0,
        )
    fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
    # plt.ylim(0, 1)
    plt.tight_layout()
    fig = fig.get_figure().savefig(vio_path)
    plt.clf()
    fig = sns.boxplot(
        data=data,
        # x="group_name",
        # y=i,
        )
    fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
    # plt.ylim(0, 1)
    plt.tight_layout()
    fig = fig.get_figure().savefig(box_path)
    plt.clf()

def main():
    infile_paths_aucscores = [
        "bio_cvec_rf_random_cval_auc.tsv",
        "bio_cvec_xg_random_cval_auc.tsv",
        "bio_distilbert_random_cval_auc.tsv",
        "bio_tfidf_rf_random_cval_auc.tsv",
        "bio_tfidf_xg_random_cval_auc.tsv",
        "bio_w2v_rf_random_cval_auc.tsv",
        "bio_w2v_xg_random_cval_auc.tsv",
    ]
    strategies = [
        "Count Vectorisation\nRandom Forest",
        "Count Vectorisation\nXGBoost",
        "DistilBERT",
        "TFIDF\nRandom Forest",
        "TFIDF\nXGBoost",
        "Word2Vec\nRandom Forest",
        "Word2Vec\nXGBoost",
    ]
    data = pd.concat([pd.read_csv(x) for x in infile_paths_aucscores], axis=1)
    data.columns = strategies
    plot_auc_scores(data)

if __name__ == "__main__":
    warn("""Paths are hardcoded and follow specific formats in a specific order.
         Code is specifically used to reproduce figures in the publication""")
    main()
