#!/usr/bin/python
# create huggingface dataset given input fasta, control and tokeniser files
import argparse
import gzip
import itertools
import os
import sys
from warnings import warn
from datasets import Dataset, DatasetDict
import pandas as pd
import screed
import torch
from datasets import load_dataset
from gensim.models import Word2Vec
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast
from utils import embed_seqs_kmers, parse_sp_tokenised, reverse_complement, split_datasets

def main():
    parser = argparse.ArgumentParser(
        description='Take fasta files, parameters and generate embedding. \
        Fasta files can be .gz. Sequences are reverse complemented by default.'
    )
    parser.add_argument('-i', '--infile_path', type=str, nargs="+",
                        help='path to fasta/gz file')
    parser.add_argument('-o', '--output_dir', type=str, default="embed/",
                        help='write embeddings to disk (DEFAULT: "embed/")')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='load existing model (DEFAULT: None)')
    parser.add_argument('-k' ,'--ksize', type=int, default=5,
                        help='set size of k-mers')
    parser.add_argument('-w' ,'--slide', type=int, default=1,
                        help='set length of sliding window on k-mers')
    parser.add_argument('-c', '--chunk', type=int, default=None,
                        help='split seqs into n-length blocks (DEFAULT: None)')
    parser.add_argument('-n' ,'--njobs', type=int, default=1,
                        help='set number of threads to use')
    parser.add_argument('-s' ,'--sample_seq', type=str, default=None,
                        help='set sample sequence to test model (DEFAULT: None)')
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
    if os.path.isfile(tmp):
        os.remove(tmp)

    if model == None:
        kmers = [embed_seqs_kmers(
            infile_path=i,
            ksize=ksize,
            slide=slide,
            rc=do_reverse_complement,
            chunk=chunk,
            outfile_path=tmp
            ) for i in infile_path]
        all_kmers = itertools.chain()
        for i in kmers:
            all_kmers = itertools.chain(all_kmers, i)
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
        Word2Vec.load(model)

    if sample_seq != None:
        print(model.wv[sample_seq])
        print(model.wv.most_similar(sample_seq, topn=10))

if __name__ == "__main__":
    main()
