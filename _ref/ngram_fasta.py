#!/usr/bin/python
# turn a genome fasta file into a table of k-mer frequencies meeting threshold
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import screed
from screed import ScreedDB
from tqdm import tqdm
from warnings import warn

def _isfile(outfile_path: str) -> bool:
    """Check if a file exists"""
    if os.path.isfile(outfile_path):
        return True
    else:
        return False

def build_kmers(sequence: str, ksize: int) -> str:
    """Generator that takes a fasta sequence and kmer size to return kmers"""
    for i in range(len(sequence) - ksize + 1):
        yield sequence[i:i + ksize], i, i + ksize

def count_kmers(kmers: list, coordinates: bool=True) -> dict:
    """Count k-mers from list and assign to k-mer:frequency"""
    counts = dict()

    if coordinates is True:
        for num, start, end in kmers:
            if not (num in counts):
                counts[num] = {"freq": 0, "coordinates": []}
            counts[num]["freq"] += 1
            counts[num]["coordinates"].append([start, end])
    else:
        for num in kmers:
            num = num[0]
            if not (num in counts):
                counts[num] = 0
            counts[num] += 1
    return counts

def seq_to_freq(filename: str, ksize: int, ngram: int=0, coordinates: bool=True,
                threshold: int=0, background: bool=False,
                outfile_path: str=None) -> pd.DataFrame:
    """Read seqs from file, convert to kmers, then to ngrams, get frequency."""
    seq_count = len([record for record in tqdm(screed.open(filename))])

    for record in tqdm(screed.open(filename), total=seq_count):
        name = record["name"]
        seq = record["sequence"]
        rc = screed.rc(seq)

        if background:
            seq = shuffle_seq(seq)
            rc = screed.rc(seq)
        seq = [x for x in build_kmers(seq, ksize)]
        if ngram > 0:
            seq = [x[0] for x in seq]
            seq = [x for x in build_kmers(seq, ngram)]
            seq = [("".join(x[0]), x[1], x[2]) for x in seq]
        rc = [x for x in build_kmers(rc, ksize)]
        if ngram > 0:
            rc = [x[0] for x in rc]
            rc = [x for x in build_kmers(rc, ngram)]
            rc = [("".join(x[0]), x[1], x[2]) for x in rc]
        seq = seq + rc
        counts = count_kmers(seq, coordinates=coordinates)
        threshold = int(float(threshold))
        if coordinates is True:
            counts = {key:val for key,val in counts.items() if val["freq"] > threshold}
            counts = pd.DataFrame(counts).T
        else:
            counts = {key:val for key,val in counts.items() if val > threshold}
            counts = pd.DataFrame(pd.Series(counts))
        yield counts

def shuffle_seq(seq: str):
    """Take a string and shuffle it randomly"""
    seq = list(seq)
    np.random.shuffle(seq)
    return "".join(seq)

def main():
    parser = argparse.ArgumentParser(
        description='Parse a fasta file and return ngram frequencies.'
    )
    parser.add_argument('fasta', type=str, help='path to fasta file')
    parser.add_argument('-b', '--background', action='store_true',
                        help='calculate background on shuffled seqs')
    parser.add_argument('-k', '--kmer_length', type=int, default=32,
                        help='k-mer length (DEFAULT: 3)')
    parser.add_argument('-n', '--ngram_length', type=int, default=1,
                        help='n-gram length (DEFAULT: 1)')
    parser.add_argument('-o', '--outfile_path', type=str, default=None,
                        help='path to output counts (overwrites existing!)')
    parser.add_argument('-t', '--threshold', type=int, default=0,
                        help='filter out frequencies below this level')

    args = parser.parse_args()
    bg = args.background
    fasta = args.fasta
    kmer = args.kmer_length
    outfile_path = args.outfile_path
    ngram = args.ngram_length
    threshold = args.threshold

    if outfile_path:
        outfile_fig = ".".join([outfile_path, "pdf"])

    if not outfile_path:
        outfile_path = ".".join([fasta, "ngrams.tsv.gz"])
        outfile_fig = ".".join([fasta, "ngrams.tsv.pdf"])
        if _isfile(outfile_path) is True:
            warn("".join(["This file exists, overwriting! : ", outfile_path]))
            os.remove(outfile_path)

    counts = seq_to_freq(fasta, kmer, ngram, True, threshold, False, outfile_path)
    counts = pd.concat([x for x in counts], axis=1)
    counts["all_coordinates"] = counts["coordinates"].values.tolist()
    coords = pd.DataFrame(counts["all_coordinates"])
    freq = pd.DataFrame(counts.freq.sum(axis=1).astype(int))
    counts = pd.merge(freq, coords, left_index=True, right_index=True)
    counts.columns = ["frequency_uncorrected", "coordinates"]
    coords_na = counts["coordinates"].tolist()
    coords_na = [[x[0] for x in y if type(x)==list] for y in coords_na]
    counts["coordinates"] = coords_na

    if bg is True:
        col = "frequency_corrected"
        print("Calculating background distribution to subtract from input seqs")
        bg = seq_to_freq(fasta, kmer, ngram, False, threshold, bg, outfile_path)
        bg = pd.concat([x for x in bg], axis=1).sum(axis=1).astype(int)
        tmp = counts["frequency_uncorrected"] - bg
        tmp = pd.DataFrame({col : tmp.loc[counts.index]})
        counts = pd.merge(tmp, counts, left_index=True, right_index=True)
    else:
        col = "frequency_uncorrected"

    counts = counts[counts[col] > threshold]
    counts[col] = counts[col].astype(int)
    bins = int(counts[col].describe().loc['max'])
    print(counts[col].describe())

    fig = counts[col].value_counts().sort_index().hist(bins=bins)
    fig.set_title("k-mer distribution")
    fig.set_xlabel("k-mer abundance")
    fig.set_ylabel("k-mers with this level of abundance")
    fig.figure.savefig(outfile_fig, dpi=300)
    counts.to_csv(outfile_path, sep="\t")

if __name__ == '__main__':
    main()
