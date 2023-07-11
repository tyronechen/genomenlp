#!/usr/bin/python
# take two sets of feature importances and plot highlighting any overlaps
import argparse
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import _cite_me

def _plot_bar(data, outfile_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    data["0_x"].plot.barh(ax=ax, color=data["0_y"])
    ax.set_title("Feature importances")
    ax.set_xlabel("Feature importance score")
    fig.tight_layout()
    fig.savefig(outfile_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
         description='Compare feature importances and highlight overlaps.'
        )
    parser.add_argument('-i1', '--infile_path_1', type=str, default=None,
                        help='path to [ csv | tsv ] file')
    parser.add_argument('-i2', '--infile_path_2', type=str, default=None,
                        help='path to [ csv | tsv ] file')
    parser.add_argument('--show_features', type=int, default=50,
                        help='number of shown feature importance (DEFAULT: 50)')
    parser.add_argument('-o', '--output_dir', type=str, default="./results_out",
                        help='specify path for output (DEFAULT: ./results_out)')

    args = parser.parse_args()

    infile_path_1 = args.infile_path_1
    infile_path_2 = args.infile_path_2
    show_features = args.show_features
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # compare
    data_1 = pd.read_csv(infile_path_1, sep="\t", index_col=0)[:show_features]
    data_2 = pd.read_csv(infile_path_2, sep="\t", index_col=0)[:show_features]

    data_1_index = list()
    data_2_index = list()
    for x in tqdm(data_1.index, desc="Aligning strings"):
        for y in data_2.index:
            if x in y or y in x:
                data_1_index.append(x)
                data_2_index.append(y)

    data_2_in_1 = list(set(data_1_index))
    data_1_in_2 = list(set(data_2_index))

    data_1_tag = pd.merge(left=data_1,
                          right=data_1.loc[data_2_in_1],
                          left_index=True,
                          right_index=True,
                          how="outer")
    data_2_tag = pd.merge(left=data_2,
                          right=data_2.loc[data_1_in_2],
                          left_index=True,
                          right_index=True,
                          how="outer")

    data_1_tag = data_1_tag.sort_values("0_x")[::-1]
    data_2_tag = data_2_tag.sort_values("0_x")[::-1]

    data_1_tag.fillna("grey", inplace=True)
    data_2_tag.fillna("grey", inplace=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_1_tag["0_y"][data_1_tag["0_y"] != "grey"] = "green"
        data_2_tag["0_y"][data_2_tag["0_y"] != "grey"] = "green"

    data_1_colourmap = data_1_tag[data_1_tag["0_y"] == "green"]["0_y"].to_dict()
    data_2_colourmap = data_2_tag[data_2_tag["0_y"] == "green"]["0_y"].to_dict()

    outfile_path = "/".join([output_dir, "scores_1.pdf"])
    _plot_bar(data_1_tag, outfile_path)
    outfile_path = "/".join([output_dir, "scores_2.pdf"])
    _plot_bar(data_2_tag, outfile_path)

if __name__ == "__main__":
    main()
    _cite_me()
