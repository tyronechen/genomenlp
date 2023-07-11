#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# parse sp tokens
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast
from utils import plot_token_dist, _cite_me

def main():
    parser = argparse.ArgumentParser(
         description='Parse SentencePiece output json file into a python object\
          usable by other modules'
        )
    parser.add_argument('tokeniser_path', type=str,
                        help='path to tokeniser.json file to load data from')
    parser.add_argument('-s', '--special_tokens', type=str, nargs="+",
                        default=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                        help='special tokens to be excluded from k-mers')
    parser.add_argument('-o', '--outfile_dir', type=str, default="./",
                        help='path to output plot directory')

    args = parser.parse_args()
    tokeniser_path = args.tokeniser_path

    if not os.path.exists(tokeniser_path):
        raise OSError("File does not exist!")

    plot_token_dist(tokeniser_path, args.special_tokens, args.outfile_dir)

if __name__ == "__main__":
    main()
    _cite_me()
