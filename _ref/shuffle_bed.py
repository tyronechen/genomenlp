#!/usr/bin/python
# shuffle seq in custom formatted bedfile
import argparse
from random import shuffle
import pandas as pd

def bootstrap_seq(seq: str, block_size: int=2):
    """Take a string and shuffle it in blocks of N length"""
    chunks, chunk_size = len(seq), block_size
    seq = [ seq[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    shuffle(seq)
    return "".join(seq)

def main():
    parser = argparse.ArgumentParser(
        description='Take custom formatted bedfile, shuffle sequences.'
    )
    parser.add_argument('-i', '--infile_path', type=str, required=True,
                        help='path to bed-like file with data')
    parser.add_argument('-o', '--outfile_path', type=str, default=None,
                        help='path to output (DEFAULT: infile_path.shuffled)')
    parser.add_argument('-c', '--col_data', type=int, default=6,
                        help='column with oligonucleotides (DEFAULT: 6)')
    args = parser.parse_args()
    col = args.col_data
    if not args.outfile_path:
        outfile_path = ".".join([args.infile_path, "shuffled"])

    data = pd.read_csv(args.infile_path, sep="\t", header=None)
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    data[col] = data[col].str.lower()
    shuf = [bootstrap_seq(x, 2) for x in data[col].tolist()]
    data[col] = shuf
    data.to_csv(outfile_path, sep="\t", header=False, index=False)

if __name__ == '__main__':
    main()
