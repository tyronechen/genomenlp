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
import pandas as pd
import screed
import torch
from datasets import load_dataset
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast
from utils import csv_to_hf, process_seqs, reverse_complement, split_datasets, _cite_me

def main():
    parser = argparse.ArgumentParser(
        description='Take control and test fasta files, tokeniser and convert \
        to HuggingFace🤗 dataset object. Fasta files can be .gz. Sequences are \
        reverse complemented by default.'
    )
    parser.add_argument('infile_path_1', type=str, help='path to fasta/gz file')
    parser.add_argument('infile_path_2', type=str, help='path to fasta/gz file')
    parser.add_argument('tokeniser_path', type=str, help='load tokeniser file')
    parser.add_argument('-o', '--outfile_dir', type=str, default="hf_out/",
                        help='write 🤗 dataset to directory as \
                        [ csv | json | parquet | dir/ ] (DEFAULT: "hf_out/")')
    parser.add_argument('-s', '--special_tokens', type=str, nargs="+",
                        default=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                        help='assign special tokens, eg space and pad tokens \
                        (DEFAULT: ["<s>", "</s>", "<unk>", "<pad>", "<mask>"])')
    parser.add_argument('-c', '--chunk', type=int, default=None,
                        help='split seqs into n-length blocks (DEFAULT: None)')
    parser.add_argument('--split_train', type=float, default=0.90,
                        help='proportion of training data (DEFAULT: 0.90)')
    parser.add_argument('--split_test', type=float, default=0.05,
                        help='proportion of testing data (DEFAULT: 0.05)')
    parser.add_argument('--split_val', type=float, default=0.05,
                        help='proportion of validation data (DEFAULT: 0.05)')
    parser.add_argument('--no_reverse_complement', action="store_false",
                        help='turn off reverse complement (DEFAULT: ON)')
    parser.add_argument('--no_shuffle', action="store_false",
                        help='turn off shuffle for data split (DEFAULT: ON)')

    args = parser.parse_args()
    infile_path_1 = args.infile_path_1
    infile_path_2 = args.infile_path_2
    tokeniser_path = args.tokeniser_path
    outfile_dir = args.outfile_dir
    special_tokens = args.special_tokens
    chunk = args.chunk
    split_train = args.split_train
    split_test = args.split_test
    split_val = args.split_val
    do_reverse_complement = args.no_reverse_complement
    do_shuffle = args.no_shuffle

    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")

    if not os.path.isdir(outfile_dir):
        os.makedirs(outfile_dir)

    # TODO: theres probably a better way to optimise this, all disk operations
    #   write seqs as sql db and read sequentially into pd df?
    #   but need to handle the conversion to huggingface dataset object also
    tmp_infile = "".join([outfile_dir, "/.data.tmp"])
    if os.path.exists(tmp_infile):
        os.remove(tmp_infile)
    process_seqs(infile_path_1, tmp_infile, rc=do_reverse_complement, chunk=chunk)

    tmp_control = "".join([outfile_dir, "/.null.tmp"])
    if os.path.exists(tmp_control):
        os.remove(tmp_control)
    process_seqs(infile_path_2, tmp_control, rc=do_reverse_complement, chunk=chunk)

    tmp_hf_out = "".join([outfile_dir, "data.hf.csv"])
    if os.path.exists(tmp_hf_out):
        os.remove(tmp_hf_out)

    csv_to_hf(tmp_infile, tmp_control, tmp_hf_out)
    os.remove(tmp_control)
    os.remove(tmp_infile)

    # configure data into a huggingface compatible dataset object
    # see https://huggingface.co/docs/datasets/access
    data = load_dataset('csv', data_files=tmp_hf_out, split="train")
    os.remove(tmp_hf_out)

    # we can tokenise separately if needed from the positive data
    # see https://huggingface.co/docs/transformers/fast_tokenizers for ref
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    if os.path.exists(tokeniser_path):
        print("\nUSING EXISTING TOKENISER:", tokeniser_path)
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
    # https://discuss.huggingface.co/t/class-labels-for-custom-datasets/15130
    dataset = dataset.class_encode_column('labels')
    # dont need this for most tasks, but keep it anyway and remove in training
    # dataset = dataset.remove_columns("token_type_ids")

    print("\nDATASET FEATURES:\n", dataset.features, "\n")
    print("CLASS NAMES:", dataset.features["labels"].names)
    print("CLASS COUNT:", dataset.features["labels"].num_classes)
    print("\nDATASET HAS FOLLOWING SPECIFICATIONS:\n", dataset)

    print("\nDATASET BEFORE SPLIT:\n", dataset)
    dataset = split_datasets(
        dataset, outfile_dir, train=split_train, test=split_test, val=split_val,
        shuffle=do_shuffle
        )
    print("\nDATASET AFTER SPLIT:\n", dataset)

    print("\nSAMPLE DATASET ENTRY:\n", dataset["train"][0], "\n")

    print("SAMPLE TOKEN MAPPING FOR FIRST 5 TOKENS IN SEQ:",)
    for i in dataset["train"][0]["input_ids"][0:5]:
        print("TOKEN ID:", i, "\t|", "TOKEN:", tokeniser.decode(i))

    # col_torch = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    # dataset.set_format(type='torch', columns=col_torch)
    # dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=1)
    # print("\nSAMPLE PYTORCH FORMATTED ENTRY:\n", next(iter(dataloader)))

if __name__ == "__main__":
    main()
    _cite_me()
