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
from utils import parse_sp_tokenised, _cite_me

def main():
    parser = argparse.ArgumentParser(
         description='Parse SentencePiece output json file into a python object\
          usable by other modules'
        )
    parser.add_argument('infile_path', type=str,
                        help='path to infile file to load data from')
    parser.add_argument('outfile_path', type=str,
                        help='path to output file to write data to')
    parser.add_argument('tokeniser_path', type=str,
                        help='path to tokeniser.json file to load data from')

    args = parser.parse_args()

    if not os.path.exists(args.tokeniser_path):
        raise OSError("File does not exist!")
    
    parse_sp_tokenised(
        infile_path=args.infile_path, 
        outfile_path=args.outfile_path,
        tokeniser_path=args.tokeniser_path, 
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
        chunksize=100, 
        columns=["idx", "feature", "labels", "input_ids", "token_type_ids", "attention_mask", "input_str"]
        )

if __name__ == "__main__":
    main()
    _cite_me()
