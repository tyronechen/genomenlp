#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# plot summary metrics associated with existing wandb runs
import argparse
import os
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from tqdm import tqdm
from utils import calculate_auc, get_run_metrics, _cite_me

def main():
    parser = argparse.ArgumentParser(
         description='Take wandb-like metrics or wandb run group ids and plot.'
        )
    parser.add_argument('-o', '--output_dir', type=str, default="./plots_out",
                        help='specify path for output (DEFAULT: ./plots_out)')
    parser.add_argument('-i', '--infile_path', type=str, default=None, nargs="+",
                        help='path to metrics.csv file(s)')
    parser.add_argument('-a', '--auc_scores', type=str, default=None, nargs="+",
                        help='path to auc_scores.csv file(s)')
    parser.add_argument('-r', '--wandb_runs', type=str, default=None, nargs="+",
                        help='wandb run id(s) to evaluate')
    parser.add_argument('-e', '--entity_name', type=str, default="",
                        help='provide wandb team name (if available).')
    parser.add_argument('-g', '--group_name', type=str, default=None, nargs="+",
                        help='provide wandb group name(s) (if desired).')
    parser.add_argument('-p', '--project_name', type=str, default="",
                        help='provide wandb project name (if available).')
    parser.add_argument('-m', '--metrics', type=str, nargs="+", default=[
                        "eval/accuracy", "eval/f1", "eval/precision",
                        "eval/recall"], help='choose metric(s) to extract \
                        (DEFAULT: ["eval/accuracy", "eval/f1", \
                        "eval/precision", "eval/recall"]')

    args = parser.parse_args()
    output_dir = args.output_dir
    infile_path = args.infile_path
    auc_scores = args.auc_scores
    wandb_runs = args.wandb_runs
    entity_name = args.entity_name
    group_name = args.group_name
    project_name = args.project_name
    metrics = args.metrics

    print("\n\nARGUMENTS:\n", args, "\n\n")

    input_args = [
        infile_path, wandb_runs, all([entity_name, project_name])
        ]
    if not any(input_args):
        raise IOError("Provide (infile_path) OR (auc_scores) OR (entity_name project_name [group_name] [wandb_runs]))")
    if any([infile_path, auc_scores]) and any([entity_name, project_name, wandb_runs]):
        warn("If input file(s) and run ids are provided, input file(s) will override runs!")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # connect to the finished runs using API
    api = wandb.Api(timeout=10000)

    # download metrics from all specified runs if not already available
    runs = None
    if not all([infile_path, auc_scores]):
        print("\nGET ALL ASSOCIATED RUNS\n", entity_name, project_name, group_name)
        if group_name == None:
            group_name = ["ungrouped"]
            runs = api.runs(path="/".join([entity_name, project_name]),)
            print("\nGET UNGROUPED RUNS FROM ENTITY AND PROJECT\n")
            if wandb_runs != None:
                runs = [api.run(path="/".join([entity_name, project_name, i]))
                        for i in wandb_runs]
                print("\nGET UNGROUPED RUNS FROM ENTITY AND PROJECT AND RUN_ID\n")
        else:
            runs = [api.runs(path="/".join([entity_name, project_name,]),
                             filters={"group": i}) for i in group_name]
            print("\nGET GROUPED RUNS FROM PROJECT AND ENTITY\n")

        for i in runs:
            for j in i:
                print(j)
    else:
        print("\nALL RUN METRICS PROVIDED, SKIP DOWNLOAD\n", infile_path, auc_scores)

    if auc_scores == None:
        print("\nCALCULATE AUC\n")
        auc = [[[calculate_auc(i, j) for i in run] for run in runs]
                 for j in tqdm(group_name)]
        auc = [x for y in [i for j in auc for i in j] for x in y]
        auc = pd.concat(auc).reset_index().drop(["index"], axis=1)
        auc.to_csv("/".join([output_dir,"auc_scores.tsv"]),index=False,sep="\t")
    else:
        auc = [pd.read_csv(x, sep="\t") for x in auc_scores]
        auc = pd.concat(auc, axis=1)

    violinplot_path = "/".join([output_dir, "metrics_violinplot.pdf"])
    boxplot_path = "/".join([output_dir, "metrics_boxplot.pdf"])
    print("\nPLOT AUC SCORES\n", violinplot_path, boxplot_path)
    fig = sns.violinplot(
        data=auc,
        x="group_name",
        y="auc",
        inner="box",
        cut=0,
        hue="class",
        split=True
        )
    fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    fig = fig.get_figure().savefig("/".join([output_dir, "auc_violinplot.pdf"]))
    plt.clf()
    fig = sns.boxplot(
        data=auc,
        x="group_name",
        y="auc",
        hue="class",
        )
    fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    fig = fig.get_figure().savefig("/".join([output_dir, "auc_boxplot.pdf"]))
    plt.clf()

    if runs != None:
        # this code was adapted directly from the wandb export API for python
        r_metrics = [[get_run_metrics(i, j) for i in runs] for j in group_name]
        r_metrics = [i for j in r_metrics for i in j]
        r_metrics = pd.concat(r_metrics).reset_index().drop(["index"], axis=1)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        r_metrics.to_csv("/".join([output_dir, "metrics.csv"]))

    # get the metrics from the run if not provided directly as a file
    if infile_path == None:
        infile_path = "/".join([output_dir, "metrics.csv"])

    if type(infile_path) is list:
    # if we give a list of existing files
        data = [pd.read_csv(i, index_col=0) for i in infile_path]
        data = dict(zip(infile_path, data))
        for i in infile_path:
            data[i]["infile_path"] = i
        data = pd.concat(data.values())
        data["summary"] = data["summary"].apply(eval)
        for i in metrics:
            data[i] = data["summary"].apply(lambda x: x[i])
    else:
    # if we use the file generated within this script
        data = pd.read_csv(infile_path, index_col=0)
        data["infile_path"] = infile_path
        data["summary"] = data["summary"].apply(eval)
        for i in metrics:
            data[i] = data["summary"].apply(lambda x: x[i])

    print("\nPLOT EVAL METRICS\n", output_dir, metrics)

    for i in metrics:
        i_ = i.replace("/", "_")
        vio_path = "".join([output_dir, "/metrics_violinplot_", i_, ".pdf"])
        box_path = "".join([output_dir, "/metrics_boxplot_", i_, ".pdf"])
        fig = sns.violinplot(
            data=data,
            x="group_name",
            y=i,
            inner="box",
            cut=0,
            )
        fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        fig = fig.get_figure().savefig(vio_path)
        plt.clf()
        fig = sns.boxplot(
            data=data,
            x="group_name",
            y=i,
            )
        fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        fig = fig.get_figure().savefig(box_path)
        plt.clf()

if __name__ == "__main__":
    main()
    _cite_me()
