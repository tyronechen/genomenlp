#!/usr/bin/python
# create huggingface dataset given input fasta, control and tokeniser files
import argparse
import gzip
import itertools
import os
import sys
from math import ceil
from random import choices, shuffle
from warnings import warn
from datasets import ClassLabel, Dataset, DatasetDict, Value
import pandas as pd
import screed
import torch
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast
from utils import reverse_complement

def dataset_to_disk(dataset: Dataset, outfile_dir: str):
    """Take a 🤗 dataset object, path as output and write files to disk"""
    if os.path.exists(outfile_dir):
        warn("".join(["Overwriting contents in directory!: ", outfile_dir]))
    dataset.to_csv("/".join([outfile_dir, "dataset.csv"]))
    dataset.to_json("/".join([outfile_dir, "dataset.json"]))
    dataset.to_parquet("/".join([outfile_dir, "dataset.parquet"]))
    dataset.save_to_disk("/".join([outfile_dir, "dataset"]))

def main():
    parser = argparse.ArgumentParser(
        description='Take control and test fasta files, tokeniser and convert \
        to HuggingFace🤗 dataset object. Fasta files can be .gz. Sequences are \
        reverse complemented by default'
    )
    parser.add_argument('infile_path', type=str, help='path to fasta/gz file')
    parser.add_argument('control_dist', type=str, help='supply control seqs')
    parser.add_argument('tokeniser_path', type=str, help='load tokeniser file')
    parser.add_argument('-o', '--outfile_dir', type=str, default="hf_out/",
                        help='write 🤗 dataset to directory as \
                        [ csv | json | parquet | dir/ ] (DEFAULT: "hf_out/")')
    parser.add_argument('-s', '--special_tokens', type=str, nargs="+",
                        default=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                        help='assign special tokens, eg space and pad tokens \
                        (DEFAULT: ["<s>", "</s>", "<unk>", "<pad>", "<mask>"])')
    parser.add_argument('--no_reverse_complement', action="store_false",
                        help='turn off reverse complement, has no effect if \
                        loading an existing tokeniser (DEFAULT: True)')

    args = parser.parse_args()
    infile_path = args.infile_path
    control_dist = args.control_dist
    tokeniser_path = args.tokeniser_path
    outfile_dir = args.outfile_dir
    special_tokens = args.special_tokens
    do_reverse_complement = args.no_reverse_complement

    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")

    # do reverse complement and negative control generation in the same loop
    seqs = dict()
    null = dict()

    # TODO: theres probably a better way to optimise this
    #   write seqs as sql db and read sequentially into pd df?
    #   but need to handle the conversion to huggingface dataset object also
    with screed.open(control_dist) as nullfile:
        for read in nullfile:
            head = read.name
            seq = read.sequence.upper()
            null[head] = seq
            if do_reverse_complement is True:
                rc = reverse_complement(seq)
                null["_".join([head, "RC"])] = rc

    with screed.open(infile_path) as seqfile:
        for read in seqfile:
            head = read.name
            seq = read.sequence.upper()
            seqs[head] = seq
            if do_reverse_complement is True:
                rc = reverse_complement(seq)
                seqs["_".join([head, "RC"])] = rc

    # TODO: can add more seq metadata into cols if needed
    seqs = pd.DataFrame({"feature": seqs}).reset_index()
    seqs.rename(columns={"index": "idx"}, inplace=True)
    seqs["labels"] = 1

    null = pd.DataFrame({"feature": null}).reset_index()
    null.rename(columns={"index": "idx"}, inplace=True)
    null["labels"] = 0

    # configure data into a huggingface compatible dataset object
    # see https://huggingface.co/docs/datasets/access
    data = pd.concat([seqs, null], axis=0).sort_index().reset_index(drop=True)
    data = Dataset.from_pandas(data)

    # we can tokenise separately if needed from the positive data
    # see https://huggingface.co/docs/transformers/fast_tokenizers for ref
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    if os.path.exists(tokeniser_path):
        print("USING EXISTING TOKENISER:", tokeniser_path)
        tokeniser = PreTrainedTokenizerFast(
            tokenizer_file=tokeniser_path,
            special_tokens=special_tokens,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
            )

    # configure dataset object to correctly assign class labels
    # see https://huggingface.co/docs/datasets/process for reference
    print("\nASSIGNING CLASS LABELS:")
    # TODO: add padding
    # https://discuss.huggingface.co/t/padding-in-datasets/2345/5
    dataset = data.map(lambda data: tokeniser(data['feature']), batched=True)
    tmp = dataset.features.copy()
    tmp["labels"] = Value('bool')
    tmp["labels"] = ClassLabel(num_classes=2, names=['negative', 'positive'])
    tmp["idx"] = Value('string')
    dataset = dataset.cast(tmp)
    print("\nDATASET FEATURES:\n", dataset.features, "\n")
    print("CLASS NAMES:", dataset.features["labels"].names)
    print("CLASS COUNT:", dataset.features["labels"].num_classes)
    print("\nDATASET HAS FOLLOWING SPECIFICATIONS:\n", dataset)

    print("\nWRITING 🤗 DATA TO DISK (OVERWRITES ANY EXISTING!) AT:\n", outfile_dir)
    if not os.path.isdir(outfile_dir):
        os.makedirs(outfile_dir)
    dataset_to_disk(dataset, outfile_dir)

    print("\nSAMPLE DATASET ENTRY:\n", dataset[0], "\n")

    print("SAMPLE TOKEN MAPPING FOR FIRST 5 TOKENS IN SEQ:",)
    for i in dataset[0]["input_ids"][0:5]:
        print("TOKEN ID:", i, "\t|", "TOKEN:", tokeniser.decode(i))

    col_torch = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    dataset.set_format(type='torch', columns=col_torch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    print("\nSAMPLE PYTORCH FORMATTED ENTRY:\n", next(iter(dataloader)))

if __name__ == "__main__":
    main()
