#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# conduct interpretation step
import argparse
import screed
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

def main():
    parser = argparse.ArgumentParser(
         description='Take a file of sequences, model, tokeniser and interpret.'
        )
    parser.add_argument('seqs_file', type=str,
                        help='path to fasta file')
    parser.add_argument('model', type=str,
                        help='path to model files from transformers/pytorch')
    parser.add_argument('tokeniser_path', type=str,
                        help='path to tokeniser.json dir to load data from')
    parser.add_argument('-o', '--output_dir', type=str, default="./vis_out",
                        help='specify path for output (DEFAULT: ./vis_out)')
    parser.add_argument('-l', '--label_names', type=str, default=None, nargs="+",
                        help='provide column with label names (DEFAULT: "").')

    args = parser.parse_args()
    seqs_file = args.seqs_file
    model = args.model
    tokeniser_path = args.tokeniser_path
    output_dir = args.output_dir
    label_names = args.label_names

    print("\n\nARGUMENTS:\n", args, "\n\n")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        model, local_files_only=True
        )
    tokeniser = AutoTokenizer.from_pretrained(
        tokeniser_path, local_files_only=True
        )
    explainer = SequenceClassificationExplainer(
        model, tokeniser, custom_labels=label_names
        )

    with screed.open(seqs_file) as infile:
        for read in tqdm(infile):
            explainer(read.sequence.upper())
            explainer.visualize("".join([output_dir, "/", read.name, ".html"]))

if __name__ == "__main__":
    main()
