#!/usr/bin/python
# plot summary metrics associated with existing wandb runs
import argparse
import os
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

def main():
    parser = argparse.ArgumentParser(
         description='Take wandb-like metrics or wandb run group ids and plot.'
        )
    parser.add_argument('-o', '--output_dir', type=str, default="./plots_out",
                        help='specify path for output (DEFAULT: ./plots_out)')
    parser.add_argument('-i', '--infile_path', type=str, default=None,
                        help='path to metrics.csv file')
    parser.add_argument('-r', '--wandb_runs', type=str, default=None, nargs="+",
                        help='wandb run id to evaluate')
    parser.add_argument('-e', '--entity_name', type=str, default="",
                        help='provide wandb team name (if available).')
    parser.add_argument('-g', '--group_name', type=str, default=None,
                        help='provide wandb group name (if desired).')
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
    wandb_runs = args.wandb_runs
    entity_name = args.entity_name
    group_name = args.group_name
    project_name = args.project_name
    metrics = args.metrics

    input_args = [infile_path, wandb_runs, group_name]
    if not any(input_args):
        raise OSError("Must provide one of infile_path/wandb_runs/group_name!")
    if input_args.count(None) <= 1:
        raise OSError("Must provide ONE of infile_path/wandb_runs/group_name!")

    # connect to the finished runs using API
    api = wandb.Api()

    # download metrics from all specified runs
    print("Get metrics from all associated runs")
    runs = None
    if entity_name != "" or project_name != "":
        if group_name == None:
            runs = api.runs(path="/".join([entity_name, project_name]),)
        else:
            runs = api.runs(path="/".join([entity_name, project_name]),
                            filters={"group": group_name})
        if wandb_runs != None:
            runs = [api.run(path="/".join([entity_name, project_name, i]))
                    for i in wandb_runs]

    if runs != None:
        # this code was adapted directly from the wandb export API for python
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
        runs_df = pd.DataFrame({
            "summary": summary_list,
            "config": config_list,
            "name": name_list
            })
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        runs_df.to_csv("/".join([output_dir, "metrics.csv"]))

    # get the metrics from the run if not provided directly as a file
    if infile_path == None:
        infile_path = "/".join([output_dir, "metrics.csv"])

    data = pd.read_csv(infile_path, index_col=0)
    data["summary"] = data["summary"].apply(eval)
    for i in metrics:
        data[i] = data["summary"].apply(lambda x: x[i])

    # TODO: write plotting functions given the above metrics

if __name__ == "__main__":
    main()
