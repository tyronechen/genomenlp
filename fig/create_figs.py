#!/usr/bin/python
# generate the figures in this paper
# publication metrics were obtained from dimensions.ai, from 2013-2022
# filtered by article and proceedings only (book chapters etc excluded)
# information accessed on 2022-09-26
# search terms for methods:
#   pytorch OR tensorflow OR keras OR wandb OR megvii OR megengine OR huggingface
# search terms for applications:
#   machine learning OR deep learning

import os
from warnings import warn
import matplotlib.pyplot as plt
import pandas as pd

def make_field_figure(data: list):
    app_method = pd.read_csv(data[1], index_col=0)
    api_method = pd.read_csv(data[4], index_col=0)
    app_method.columns = ["Application"]
    api_method.columns = ["High level API"]
    app_api_method = app_method.join(api_method, how="outer")
    app_api_method.sort_values(by="Application", ascending=True, inplace=True)
    app_api_method.plot(kind="barh", figsize=(10,6))
    plt.tight_layout()
    plt.savefig("app_api_year.pdf", dpi=300)
    plt.savefig("app_api_year.png", dpi=300)
    plt.close()

def make_publication_figure(data: list):
    app_pub = pd.read_csv(data[2], index_col=0)
    api_pub = pd.read_csv(data[5], index_col=0)
    app_cit = pd.read_csv(data[0], index_col=0)
    api_cit = pd.read_csv(data[3], index_col=0)
    app_pub.columns = ["Application (publications)"]
    api_pub.columns = ["High level API (publications)"]
    app_cit.columns = ["Application (citations)"]
    api_cit.columns = ["High level API (citations)"]
    # app_api = pd.DataFrame().join([app_pub, api_pub, app_cit, api_cit], how="outer")
    app_api = app_pub.join(api_cit, how="outer")
    app_api.plot(kind="barh", figsize=(10,6))
    plt.tight_layout()
    plt.savefig("app_api_pub.pdf", dpi=300)
    plt.savefig("app_api_pub.png", dpi=300)
    plt.close()

def main():
    infile_paths_publications = [
        "application_chart_citation.csv",
        "application_chart_field.csv",
        "application_chart_year.csv",
        "methods_chart_citation.csv",
        "methods_chart_field.csv",
        "methods_chart_year.csv",
        ]
    make_field_figure(infile_paths_publications)
    make_publication_figure(infile_paths_publications)

if __name__ == "__main__":
    warn("""Paths are hardcoded and follow specific formats in a specific order.
         Code is specifically used to reproduce figures in the publication""")
    main()
