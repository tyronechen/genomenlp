#!/usr/bin/python
# main controller script for running the pipeline
import argparse
import json
import os
import sys
from warnings import warn

class _DictToClass:
    # this is just to convert json entries to an argparse-like object
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main():
    parser = argparse.ArgumentParser(
        description='Generates a pipeline script based on command line input.'
    )
    parser.add_argument('-j', '--json_path', type=str,
                        help='path to json file with all input parameters')
    parser.add_argument('-i', '--infile_path', type=str,
                        help='path to fasta file with data')
    parser.add_argument('-o', '--outfile_dir', type=str, default="out/",
                        help='path to output directory (DEFAULT: out/)')
    parser.add_argument('-c', '--control', type=str, default=None,
                        help='[ /path/to/file | bootstrap | frequency ]')
    parser.add_argument('-b', '--block_size', type=int, default=2,
                        help='size of block to generate null seqs from, \
                        only if generating controls (DEFAULT: 2)')
    parser.add_argument('--no_reverse_complement', action="store_false",
                        help='turn off reverse complement, only if \
                        generating controls (DEFAULT: True)')
    parser.add_argument('-t', '--tokeniser_path', type=str, default="",
                        help='path to tokeniser.json file to save or load data')
    parser.add_argument('-r', '--representation', type=str, default=None,
                        help='[ dna2vec | ngram | sentencepiece ]')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='[ neuralnet | randomforest | xgboost ]')
    args = parser.parse_args()

    if args.infile_path is None and args.json_path is None:
        raise OSError("Input data must be provided either as a path on the \
                      command line or as a path in the json file")

    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")

    if args.json_path:
        warn("json file is provided, overriding all command line arguments!")
        with open(args.json_path, mode="r") as j:
            args = json.load(j)
            args = _DictToClass(**args)

    infile_path = args.infile_path
    outfile_dir = args.outfile_dir
    control = args.control
    block_size = args.block_size
    no_reverse_complement = args.no_reverse_complement
    tokeniser_path = args.tokeniser_path
    representation = args.representation
    model = args.model

    if not os.path.isdir(outfile_dir):
        os.makedirs(outfile_dir)

    if control == "bootstrap" or control == "frequency":
        if os.path.isfile(control):
            raise OSError("Why are you naming your control distribution file \
                          the same as a command line argument? (╯°□°)╯︵ ┻━┻")

    if os.path.isfile(control):
        warn("Since control is provided as a file of sequences, command line \
             arguments block_size and no_reverse_complement are ignored.")

if __name__ == "__main__":
    main()
