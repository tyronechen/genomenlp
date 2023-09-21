#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# perform power law fitting
import argparse
import os
import logging
import sys
from warnings import warn
import matplotlib.pyplot as plt
from transformers import \
    AutoModel, BertModel, DistilBertModel, RobertaModel, XLNetModel
from utils import plot_hist, plot_scatter, _cite_me
import wandb
import weightwatcher as ww

def main():
    parser = argparse.ArgumentParser(
        description='Take trained model dataset and apply power law fit. \
        Acts as a performance metric which is independent of data. \
        For more information refer here: https://arxiv.org/pdf/2202.02842.pdf'
        )
    parser.add_argument('model_path', type=str, nargs="+",
                        help='path to trained model online configureation or local directory (local has priority!)')
    parser.add_argument('-o', '--output_dir', type=str, default="./powerlaw_out",
                        help='path to output metrics directory \
                        (DEFAULT: same as model_path)')
    parser.add_argument('-a', '--alpha_max', type=int, default=8,
                        help='max alpha value to plot (DEFAULT: 8)')
    args = parser.parse_args()
    model_path = args.model_path
    output_dir = args.output_dir
    alpha_max = args.alpha_max

    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")

    if model_path == None:
        raise OSError("Must provide valid paths to model file(s)!")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        warn("".join(["Overwriting files in: ", output_dir]))

    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger('weightwatcher')
    # logger.setLevel(logging.INFO)
    # warnings.filterwarnings('ignore')

    model_path_final = list()

    if model_path != None:
        for i in model_path:
            if os.path.exists(i):
                print("Use model files from:", i)
                model_path_final.append(i)
                warn("If you are intending to download a model and the directory path matches the one on your disk, you will need to rename or remove it since it will first use local files as a priority!")
            else:
                print("Downloading model files from wandb:", i)
                api = wandb.Api(timeout=10000)
                run = api.run(path=i)
                for j in run.files():
                    i = "/".join([output_dir, os.path.basename(i)])
                    j.download(root=i, replace=True) 
                print("Loading model from:", i)
                model = AutoModel.from_pretrained(i)
                watcher = ww.WeightWatcher(model=model)
                details = watcher.describe()
                print("\nMODEL_DETAILS (summary):\n")
                print(details)
                model_out = "/".join([output_dir, os.path.basename(i)])
                model_path_final.append(model_out)
                details = watcher.analyze(
                    randomize=True, min_evals=50, plot=True, savefig=model_out
                    )
                print("\nMODEL_DETAILS (with fit):\n")
                print(details)
                alpha_main = "/".join([model_out, "alpha_main.tsv"])
                details.to_csv(alpha_main, sep="\t")
                model_info = details[(details.alpha<alpha_max) & (details.alpha>0)]   

    alpha_hist = "/".join([output_dir, "alpha_hist.pdf"])
    alpha_plot = "/".join([output_dir, "alpha_plot.pdf"])

    # save all the trained weights for further reuse and write to output_dir
    if model_path != None:
        print("\nCOMPARISONS:\n", model_path, "\n")
        model_path = [(i,ww.WeightWatcher(AutoModel.from_pretrained(i)).analyze(
                        randomize=True, min_evals=50)) for i in model_path_final]
        for i, j in model_path:
            out_dir = "".join([output_dir, "/", i.split("/")[-1], "/"])
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            j.to_csv("".join([out_dir, "alpha.tsv"]),sep="\t")
            plot_hist([(i, j)], "".join([out_dir, "alpha_hist.pdf"]))
            plot_scatter([(i, j)], "".join([out_dir, "alpha_plot.pdf"]))

    plot_hist(model_path, alpha_hist)
    plot_scatter(model_path, alpha_plot)

if __name__ == "__main__":
    main()
    _cite_me()
