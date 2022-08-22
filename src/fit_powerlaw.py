import argparse
import os
import logging
import sys
from warnings import warn
import matplotlib.pyplot as plt
from transformers import \
    AutoModel, BertModel, DistilBertModel, RobertaModel, XLNetModel
from utils import plot_hist, plot_scatter
import weightwatcher as ww

def main():
    parser = argparse.ArgumentParser(
        description='Take trained model dataset and apply power law fit. \
        Acts as a performance metric which is independent of data. \
        For more information refer here: https://arxiv.org/pdf/2202.02842.pdf'
        )
    parser.add_argument('model_path', type=str,
                        help='path to trained model directory')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='path to output metrics directory \
                        (DEFAULT: same as model_path)')
    parser.add_argument('-c', '--compare_to', type=str, nargs="+", default=None,
                        help='path(s) to models to compare (DEFAULT: None)')
    parser.add_argument('-a', '--alpha_max', type=int, default=8,
                        help='max alpha value to plot (DEFAULT: 8)')
    args = parser.parse_args()
    model_path = args.model_path
    output_dir = args.output_dir
    compare_to = args.compare_to
    alpha_max = args.alpha_max

    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")

    if output_dir == None:
        output_dir = "/".join([model_path, "fit_powerlaw"])
        print("No output_dir provided, default to:", output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        warn("".join(["Overwriting files in: ", output_dir]))

    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger('weightwatcher')
    # logger.setLevel(logging.INFO)
    # warnings.filterwarnings('ignore')

    model = AutoModel.from_pretrained(model_path)
    watcher = ww.WeightWatcher(model=model)
    details = watcher.describe()
    print("\nMODEL_DETAILS (summary):\n")
    print(details)
    details = watcher.analyze(
        randomize=True, min_evals=50, plot=True, savefig=output_dir
        )
    print("\nMODEL_DETAILS (with fit):\n")
    print(details)

    alpha_main = "/".join([output_dir, "alpha_main.tsv"])
    alpha_hist = "/".join([output_dir, "alpha_hist.pdf"])
    alpha_plot = "/".join([output_dir, "alpha_plot.pdf"])

    details.to_csv(alpha_main, sep="\t")

    model_info = details[(details.alpha < alpha_max) & (details.alpha > 0)]

    # save all the trained weights for further reuse and write to output_dir
    if compare_to != None:
        print("\nCOMPARISONS:\n", compare_to, "\n")
        compare_to = [(i,ww.WeightWatcher(AutoModel.from_pretrained(i)).analyze(
                        randomize=True, min_evals=50)) for i in compare_to]
        for i, j in compare_to:
            j.to_csv("".join([output_dir,"/",i.split("/")[-1],".tsv"]),sep="\t")

    plot_hist(model_info, alpha_hist, compare_to)
    plot_scatter(model_info, alpha_plot, compare_to)

if __name__ == "__main__":
    main()
