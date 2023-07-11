#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# create huggingface dataset given input fasta, control and tokeniser files
import argparse
from collections import Counter
import gzip
import itertools
import os
import sys
from warnings import warn
from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd
import screed
import torch
from datasets import load_dataset
from gensim.models import Word2Vec
from tokenizers import SentencePieceUnigramTokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from utils import build_kmers, embed_seqs_kmers, embed_seqs_sp, parse_sp_tokenised, reverse_complement, split_datasets, _cite_me

def parse_kmers(infile_path, ksize, slide, colname="feature"):
    """Generator to parse kmers from a SP-like tokenised data file"""
    for j in tqdm(
        pd.read_csv(infile_path, index_col=0, chunksize=1),desc="Extract tokens"
        ):
        seq = j[colname].iloc[0]
        yield [seq[i:i + ksize] for i in range(len(seq) - ksize + 1)][::slide]

def main():
    parser = argparse.ArgumentParser(
        description='Take tokenised data, parameters and generate embedding. \
        Note that this takes output of kmerise_bio.py, and NOT raw fasta files.'
    )
    parser.add_argument('-i', '--infile_path', type=str, nargs="+",
                        help='path to input tokenised data file')
    parser.add_argument('-o', '--output_dir', type=str, default="embed/",
                        help='write embeddings to disk (DEFAULT: "embed/")')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='load existing model (DEFAULT: None)')
    parser.add_argument('-k' ,'--ksize', type=int, default=5,
                        help='set size of k-mers')
    parser.add_argument('-w' ,'--slide', type=int, default=1,
                        help='set length of sliding window on k-mers (min 1)')
    parser.add_argument('-c', '--chunk', type=int, default=None,
                        help='split seqs into n-length blocks (DEFAULT: None)')
    parser.add_argument('-n' ,'--njobs', type=int, default=1,
                        help='set number of threads to use')
    parser.add_argument('-s' ,'--sample_seq', type=str, default=None,
                        help='set sample sequence to test model (DEFAULT: None)')
    parser.add_argument('-v', '--vocab_size', type=int, default=0,
                        help='vocabulary size for model config (DEFAULT: all)')
    parser.add_argument('--w2v_min_count' ,type=int, default=1,
                        help='set minimum count for w2v (DEFAULT: 1)')
    parser.add_argument('--w2v_sg' ,type=int, default=1,
                        help='0 for bag-of-words, 1 for skip-gram (DEFAULT: 1)')
    parser.add_argument('--w2v_vector_size' ,type=int, default=100,
                        help='set w2v matrix dimensions (DEFAULT: 100)')
    parser.add_argument('--w2v_window' ,type=int, default=10,
                        help='set context window size for w2v (DEFAULT: -/+10)')
    parser.add_argument('--no_reverse_complement', action="store_false",
                        help='turn off reverse complement (DEFAULT: ON)')

    args = parser.parse_args()
    infile_path = args.infile_path
    output_dir = args.output_dir
    model = args.model
    ksize = args.ksize
    slide = args.slide
    chunk = args.chunk
    njobs = args.njobs
    vocab_size = args.vocab_size
    w2v_min_count = args.w2v_min_count
    w2v_sg = args.w2v_sg
    w2v_window = args.w2v_window
    w2v_vector_size = args.w2v_vector_size
    do_reverse_complement = args.no_reverse_complement
    sample_seq = args.sample_seq

    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    tmp = "/".join([output_dir, ".tmp"])
    model_path = "/".join([output_dir, "kmers.model"])
    vector_path = "/".join([output_dir, "kmers.w2v"])
    tokens_path = "/".join([output_dir, "kmers.csv"])
    projected_path = "/".join([output_dir, "kmers_embeddings.csv"])
    if os.path.isfile(tmp):
        os.remove(tmp)

    if model == None:
        kmers = [parse_kmers(i, ksize, slide) for i in infile_path]
        all_kmers = itertools.chain()
        for i in kmers:
            all_kmers = itertools.chain(all_kmers, i)

        if vocab_size > 0:
            # reduce vocab_size to desired amount
            counts = Counter()
            for i in all_kmers:
                counts += Counter(i)
            counts = {
                k: v for k, v in
                sorted(counts.items(), key=lambda item: item[1], reverse=True)
                }
            counts = list(counts.keys())[:vocab_size]

            # reset generator as it was consumed in previous step
            kmers = [parse_kmers(i, ksize, slide) for i in infile_path]
            all_kmers = itertools.chain()
            for i in kmers:
                all_kmers = itertools.chain(all_kmers, i)

            all_kmers = [
                sentence for sentence in [
                    [
                        word for word in sequence if word in counts
                        ] for sequence in all_kmers
                    ] if len(sentence) > 0
                ]

        model = Word2Vec(
            sentences=all_kmers,
            vector_size=w2v_vector_size,
            window=w2v_window,
            min_count=w2v_min_count,
            sg=w2v_sg,
            workers=njobs
            )
        model.save(model_path)
        model.wv.save(vector_path)
    else:
        model = Word2Vec.load(model)

    if sample_seq != None:
        print(model.wv[sample_seq])
        print(model.wv.most_similar(sample_seq, topn=10))

    # NOTE: different from SP pathway! here kmerise_bio.py gets tokens directly
    # so we just plug that output into here - data files not fasta files
    # then map these onto embedding into projection vectors
    if os.path.isfile(projected_path):
        os.remove(projected_path)

    for i in infile_path:
        for entry in tqdm(
            pd.read_csv(i, index_col=0, chunksize=1), desc="Extract tokens"
            ):
            # tokens = entry["input_str"].apply(
            #     lambda x: x[1:-1].replace("\'", "").split()
            #     ).iloc[0]
            seq = entry["feature"].iloc[0]
            # meta = pd.DataFrame(
            #     {"labels": [entry["labels"].iloc[0]],
            #      "seq": "".join(tokens[::ksize])}
            #     )
            if len(seq) == 0:
                pass
            else:
                tokens = [seq[i:i+ksize] for i in range(len(seq)-ksize+1)][::slide]
                if len(tokens) == 0:
                    pass
                else:
                    if vocab_size > 0:
                        tokens = [x for x in tokens if x in counts]
                    if len(tokens) == 0:
                        pass
                    else:
                        data = pd.DataFrame(model.wv[tokens])
                        data["labels"] = entry["labels"].iloc[0]
                        data["seq"] = tokens
                        cols = data.columns.tolist()
                        data = data[cols[-2:] + cols[:-2]]
                        data.to_csv(
                            projected_path, mode="a+", header=False, index=False
                            )
                    # data = pd.DataFrame(np.concatenate(model.wv[tokens])).transpose()
                    # data = pd.concat([meta, data], axis=1)
                    # data.to_csv(projected_path, mode="a+", header=False, index=False)

if __name__ == "__main__":
    main()
    _cite_me()
