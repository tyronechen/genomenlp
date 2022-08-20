import argparse
import os
import logging
import sys
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
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger('weightwatcher')
    # logger.setLevel(logging.INFO)
    # warnings.filterwarnings('ignore')

    model = AutoModel.from_pretrained(model_path)
    watcher = ww.WeightWatcher(model=model)
    details = watcher.describe()
    print("\nMODEL_DETAILS (summary):\n")
    print(details)
    details = watcher.analyze(randomize=True, min_evals=50)
    print("\nMODEL_DETAILS (with fit):\n")
    print(details)

    alpha_list = "/".join([output_dir, "alpha_list.tsv"])
    details.to_csv(alpha_list, sep="\t")

    alpha_hist = "/".join([output_dir, "alpha_hist.pdf"])
    alpha_plot = "/".join([output_dir, "alpha_plot.pdf"])

    model_info = details[(details.alpha < alpha_max) & (details.alpha > 0)]
    print("\nCOMPARISONS:", compare_to, "\n")
    plot_hist(model_info, alpha_hist, compare_to)
    plot_scatter(model_info, alpha_plot, compare_to)

    # model_info.alpha.plot.hist(bins=100, label='main', density=True, color='blue')
    # plt.axvline(model_info.alpha.mean(), color='blue', linestyle='dashed')
    # plt.legend()
    # plt.savefig(alpha_hist, dpi=300)
    # plt.close()

    # x = model_info.layer_id.to_numpy()
    # y = model_info.alpha.to_numpy()
    # plt.scatter(x, y, color='blue')
    # plt.axhline(np.mean(y), color='blue', linestyle='dashed')
    # plt.savefig(alpha_plot, dpi=300)
    # plt.close()

if __name__ == "__main__":
    main()
