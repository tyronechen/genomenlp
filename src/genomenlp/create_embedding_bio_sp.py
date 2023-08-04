#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# create huggingface dataset given input fasta, control and tokeniser files
import argparse
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
from transformers import PreTrainedTokenizerFast
from utils import embed_seqs_kmers, embed_seqs_sp, parse_sp_tokenised, reverse_complement, split_datasets, _cite_me

def main():
    parser = argparse.ArgumentParser(
        description='Take fasta files, tokeniser and generate embedding. \
        Fasta files can be .gz. Sequences are reverse complemented by default.'
    )
    parser.add_argument('-i', '--infile_path', type=str, nargs="+",
                        help='path to fasta/gz file')
    parser.add_argument('-o', '--output_dir', type=str, default="embed/",
                        help='write embeddings to disk (DEFAULT: "embed/")')
    parser.add_argument('-c', '--column_names', type=str, default=["idx",
                        "feature", "labels", "input_ids", "token_type_ids",
                        "attention_mask", "input_str"],
                        help='column name for sp tokenised data \
                        (DEFAULT: ["idx", "feature", "labels", \
                        "input_ids", "token_type_ids", \
                        "attention_mask", "input_str"])')
    parser.add_argument('-l', '--labels', type=str, default="labels",
                        help='column name for data labels (DEFAULT: labels)')
    parser.add_argument('-x', '--column_name', type=str, default="input_str",
                        help='column name for extracting embeddings \
                        (DEFAULT: input_str)')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='load existing model (DEFAULT: None)')
    parser.add_argument('-t' ,'--tokeniser_path', type=str,
                        help='load tokenised data')
    parser.add_argument('-s', '--special_tokens', type=str, nargs="+",
                        default=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                        help='assign special tokens, eg space and pad tokens \
                        (DEFAULT: ["<s>", "</s>", "<unk>", "<pad>", "<mask>"])')
    parser.add_argument('-n' ,'--njobs', type=int, default=1,
                        help='set number of threads to use')
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
    parser.add_argument('--sample_seq' ,type=str, default=None,
                        help='project sample sequence on embedding (DEFAULT: None)')

    args = parser.parse_args()
    infile_path = args.infile_path
    output_dir = args.output_dir
    labels = args.labels
    column_name = args.column_name
    column_names = args.column_names
    model = args.model
    tokeniser_path = args.tokeniser_path
    special_tokens = args.special_tokens
    njobs = args.njobs
    w2v_min_count = args.w2v_min_count
    w2v_sg = args.w2v_sg
    w2v_window = args.w2v_window
    w2v_vector_size = args.w2v_vector_size
    sample_seq = args.sample_seq
    do_reverse_complement = args.no_reverse_complement

    # infile_path = ["../results/tmp/bcwp_kmer_5/train.csv"]
    # output_dir = "../results/tmp/embed_bcwp_kmers"
    # labels = "labels"
    # column_name = "input_str"
    # column_names = ["idx", "feature", "labels", "input_ids", "token_type_ids", "attention_mask", "input_str"]
    # tokeniser_path = "../data/prot/BCW_prot_kmers.json"
    # special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    # njobs = 1
    # w2v_min_count = 1
    # w2v_sg = 1
    # w2v_window = 100
    # w2v_vector_size = 10

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
        # [parse_sp_tokenised(i, tokens_path, tokeniser_path, special_tokens)
        #  for i in infile_path]
        kmers = [embed_seqs_sp(
            infile_path=i,
            outfile_path=tokens_path,
            chunksize=1,
            tokeniser_path=tokeniser_path,
            special_tokens=special_tokens,
            columns=column_names,
            column=column_name,
            labels=labels,
            ) for i in infile_path]
        tmp_kmers = itertools.chain()
        for i in kmers:
            tmp_kmers = itertools.chain(tmp_kmers, i)        
        all_kmers = itertools.chain()
        for i, j in tmp_kmers:
            all_kmers = itertools.chain(all_kmers, [i])
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

    kmers = [embed_seqs_sp(
        infile_path=i,
        outfile_path=tokens_path,
        chunksize=1,
        tokeniser_path=tokeniser_path,
        special_tokens=special_tokens,
        columns=column_names,
        column=column_name,
        labels=labels
        ) for i in infile_path]
    all_kmers = itertools.chain()
    for i in kmers:
        all_kmers = itertools.chain(all_kmers, i)
    if os.path.isfile(projected_path):
        os.remove(projected_path)
    for i, j in all_kmers:
        data = pd.DataFrame(model.wv[i])
        data["labels"] = j
        data["seq"] = i
        cols = data.columns.tolist()
        data = data[cols[-2:] + cols[:-2]]
        data.to_csv(projected_path, mode="a+", header=False, index=False)
        # meta = pd.DataFrame({"labels": [j], "seq": "".join(i)})
        # data = pd.DataFrame(np.concatenate(model.wv[i])).transpose()
        # data = pd.concat([meta, data], axis=1)
        # data.to_csv(projected_path, mode="a+", header=False, index=False)

if __name__ == "__main__":
    main()
    _cite_me()
