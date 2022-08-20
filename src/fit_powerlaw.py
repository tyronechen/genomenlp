import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import \
    TFAutoModel, BertModel, DistilBertModel, RobertaModel, XLNetModel
import weightwatcher as ww

def main():
    parser = argparse.ArgumentParser(
        description='Take trained model dataset and apply power law fit. \
        Acts as a performance metric which is independent of data. \
        For more information refer here: https://arxiv.org/pdf/2202.02842.pdf
        )
    parser.add_argument('model_path', type=str,
                        help='path to trained model directory')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='path to output metrics directory \
                        (DEFAULT: same as model_path)')
    parser.add_argument('-c', '--compare_to', type=str, nargs="+", default=None,
                        help='compare model performance to others [ BERT | \
                        DistilBERT | RoBERTa | XLNet ] (DEFAULT: None)')
    parser.add_argument('-a', '--alpha_max', type=int, default=8,
                        help='max alpha value to plot (DEFAULT: 8)')
    args = parser.parse_args()
    model_path = args.model_path
    output_dir = args.output_dir
    alpha_max = args.alpha_max

    if output_dir == None:
        output_dir = "/".join([model_path, "fit_powerlaw"])
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    model = TFAutoModel.from_pretrained(model_path, from_pt=True)
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
    model_main = details[(details.alpha < alpha_max) & (details.alpha > 0)]

    model_main.alpha.plot.hist(bins=100, label='main', density=True, color='blue')
    plt.axvline(model_main.alpha.mean(), color='blue', linestyle='dashed')
    plt.legend()
    plt.savefig(alpha_hist, dpi=300)
    plt.close()

    alpha_plot = "/".join([output_dir, "alpha_plot.pdf"])
    x = model_main.layer_id.to_numpy()
    y = model_main.alpha.to_numpy()
    plt.scatter(x, y, color='blue')
    plt.axhline(np.mean(y), color='blue', linestyle='dashed')
    plt.savefig(alpha_plot, dpi=300)
    plt.close()

    distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    roberta = RobertaModel.from_pretrained('roberta-base-uncased')
    xlnet = XLNetModel.from_pretrained('xlnet-base-uncased')
