#!/usr/bin/python
# tokenise fasta sequences with sentencepiece and export as json file
import argparse
import gzip
import json
import os
import sys
from warnings import warn
import numpy as np
import pandas as pd
import screed
from tokenizers import SentencePieceUnigramTokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from utils import build_kmers, _init_sp_tokeniser

def main():
    parser = argparse.ArgumentParser(
        description='Take gzip fasta file(s), kmerise reads and export csv.'
    )
    parser.add_argument('-i', '--infile_path', type=str, default=None, nargs="+",
                        help='path to files with biological seqs split by line')
    parser.add_argument('-o', '--outfile_path', type=str, default="out.csv",
                        help='path to output huggingface-like dataset.csv file')
    parser.add_argument('-m', '--mappings', type=str, default="mappings.json",
                        help='path to output mappings file')
    parser.add_argument('-t', '--tokeniser_path', type=str, default="tokeniser.json",
                        help='path to tokeniser.json file to save data')
    parser.add_argument('-k', '--kmer_size', type=int, default=0,
                        help='split seqs into n-length blocks (DEFAULT: None)')
    parser.add_argument('-l', '--label', type=int, default=None, nargs="+",
                        help='provide integer label for seqs \
                        (order must match infile_path!)')
    parser.add_argument('--no_reverse_complement', action="store_false",
                        help='turn off reverse complement (DEFAULT: ON)')

    args = parser.parse_args()
    infile_path = args.infile_path
    outfile_path = args.outfile_path
    tempfile_path = "".join([
        os.path.dirname(outfile_path), "/.",
        os.path.basename(outfile_path), ".tmp"
        ])
    kmer_size = args.kmer_size
    label = args.label
    mappings = args.mappings
    tokeniser_path = args.tokeniser_path
    do_reverse_complement = args.no_reverse_complement
    np.set_printoptions(threshold=sys.maxsize)

    if infile_path == None:
        raise OSError("Provide either input fasta file or existing tokeniser!")
    if len(infile_path) != len(label):
        raise OSError("infile_path:label combinations must match")
    infile_label = list(zip(infile_path, label))

    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")

    if os.path.exists(outfile_path):
        warn("Overwriting existing output file in --outfile_path!")
        os.remove(outfile_path)
    if os.path.exists(tempfile_path):
        os.remove(tempfile_path)

    header = \
      ",idx,feature,labels,input_ids,input_str,token_type_ids,attention_mask\n"
    with open(tempfile_path, mode="a+") as tempfile:
        tempfile.write(header)
        for i, j in infile_label:
            with screed.open(i) as infile:
                count = 0
                for read in tqdm(infile, desc="Parsing reads"):
                    idx = read.name
                    feature = read.sequence
                    labels = str(j)
                    input_ids = str(np.nan)
                    input_arr = np.array(
                        [i for i in build_kmers(read.sequence, kmer_size)]
                        )
                    input_str = str(input_arr)
                    token_type_ids = str(np.zeros(len(input_arr), dtype=int))
                    attention_mask = str(np.ones(len(input_arr), dtype=int))
                    data = "".join([
                        str(count), ",",
                        "\"", idx, "\",",
                        "\"", feature, "\",",
                        labels, ",",
                        "\"", input_ids, "\",",
                        "\"", input_str, "\",",
                        "\"", token_type_ids, "\",",
                        "\"", attention_mask, "\"",
                    ])
                    tempfile.write(data + "\n")
                    count += 1

    tempfile = pd.read_csv(tempfile_path, index_col=0, sep=",", chunksize=1)
    unique = set()
    for data in tqdm(tempfile, desc="Building vocabulary"):
        keys = data["input_str"].iloc[0][
            1:-1
            ].replace("\n", "").replace("\'", "").split(" ")
        unique |= set(keys)

    vocab_key = dict()
    for i, j in enumerate(unique):
        vocab_key[j] = i
    with open(mappings, mode="w") as args_json:
        json.dump(vocab_key, args_json, ensure_ascii=False, indent=4)

    tokeniser = _init_sp_tokeniser(unique)
    with open(tokeniser_path, mode="w") as token_out:
        json.dump(tokeniser_path, token_out, ensure_ascii=False, indent=4)

    with open(outfile_path, mode="a+") as outfile:
        tempfile = pd.read_csv(tempfile_path, index_col=0, sep=",", chunksize=1)
        for data in tqdm(tempfile, desc="Mapping vocabulary"):
            input_str = pd.Series(np.array(
                data["input_str"].iloc[0][1:-1].replace("\'", "").split()
                ))
            data["input_ids"] = str(input_str.apply(
                lambda x: np.vectorize(vocab_key.get)(x)
                ).to_list())
            data.to_csv(outfile_path, header=False, mode="a+")

if __name__ == "__main__":
    main()
