#!/usr/bin/python
# turn a genome fasta file into a table of k-mer frequencies meeting threshold
import argparse
import numpy as np
import pandas as pd
import screed

def build_kmers(sequence: str, ksize: int) -> str:
    """Generator that takes a fasta sequence and kmer size to return kmers"""
    for i in range(len(sequence) - ksize + 1):
        yield sequence[i:i + ksize]

def read_kmers(filename: str, ksize: int) -> list:
    """Read k-mers from file and get k-mer frequency table"""
    all_kmers = list()
    for record in screed.open(filename):
        all_kmers += build_kmers(record.sequence, ksize)
        yield all_kmers

def count_kmers(kmers: int) -> dict:
    """Count k-mers from list and assign to k-mer:frequency"""
    counts = dict()
    for num in kmers:
        if not (num in counts):
            counts[num] = 0
        counts[num] += 1
    return counts

def write_counts(counts: dict, outfile_path: str=None, rows: int=1) -> pd.DataFrame:
    """Export counts to tsv file"""
    counts = {key:val for key, val in counts.items() if val != 1}
    counts = pd.DataFrame(pd.Series(counts))
    counts = counts[counts[0] > counts[0].describe()[6]]
    counts.columns = ["kmers"]
    counts = counts.T
    columns = counts.columns
    counts = pd.DataFrame(np.repeat(counts.values, rows, axis=0))
    counts.columns = columns
    counts.index = ["kmers"] * rows
    if outfile_path:
        print("Writing file to:", outfile_path)
        counts.to_csv(outfile_path, sep="\t")
    return counts

def main():
    parser = argparse.ArgumentParser(
        description='Parse a fasta file and return thresholded k-mer counts.'
    )
    parser.add_argument('fasta', type=str, help='path to fasta file')
    parser.add_argument('-k', '--kmer_length', type=int, default=32,
                        help='k-mer length (DEFAULT: 32)')
    parser.add_argument('-r', '--rows', type=int, default=1,
                        help='replicate rows along the dataframe (DEFAULT: 1)')
    parser.add_argument('-o', '--outfile_path', type=str, default=None,
                        help='path to output counts (overwrites existing!)')

    args = parser.parse_args()

    fasta = args.fasta
    k = args.kmer_length
    outfile_path = args.outfile_path
    rows = args.rows
    if not outfile_path:
        outfile_path = ".".join([fasta, "tsv.gz"])

    data = [i for i in read_kmers(fasta, k)]
    data = count_kmers(data[0])
    data = write_counts(data, outfile_path, rows)

if __name__ == '__main__':
    main()
