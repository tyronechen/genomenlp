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
from datasets import load_dataset
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast
from utils import reverse_complement

def dataset_to_disk(dataset: Dataset, outfile_dir: str, name: str):
    """Take a 🤗 dataset object, path as output and write files to disk"""
    if os.path.exists(outfile_dir):
        warn("".join(["Overwriting contents in directory!: ", outfile_dir]))
    dataset.to_csv("".join([outfile_dir, "/", name, ".csv"]))
    dataset.to_json("".join([outfile_dir, "/", name, ".json"]))
    dataset.to_parquet("".join([outfile_dir, "/", name, ".parquet"]))
    dataset.save_to_disk("".join([outfile_dir, "/", name]))

def split_datasets(dataset: DatasetDict, outfile_dir: str, train: float,
                   test: float=0, val: float=0, shuffle: bool=False):
    """Split data into training | testing | validation sets"""
    assert train + test + val == 1, "Proportions of datasets must sum to 1!"
    train_split = 1 - train
    test_split = 1 - test / (test + val)
    val_split = 1 - val / (test + val)

    train = dataset.train_test_split(test_size=train_split, shuffle=shuffle)
    if val > 0:
        test_valid = train['test'].train_test_split(test_size=test_split, shuffle=shuffle)
        data = DatasetDict({
            'train': train['train'],
            'test': test_valid['test'],
            'valid': test_valid['train'],
            })
        dataset_to_disk(data["train"], outfile_dir, "train")
        dataset_to_disk(data["test"], outfile_dir, "test")
        dataset_to_disk(data["valid"], outfile_dir, "valid")
        return data
    else:
        data = DatasetDict({
            'train': train['train'],
            'test': train['test'],
            })
        dataset_to_disk(data["train"], outfile_dir, "train")
        dataset_to_disk(data["test"], outfile_dir, "test")
        return data

def main():
    parser = argparse.ArgumentParser(
        description='Take control and test fasta files, tokeniser and convert \
        to HuggingFace🤗 dataset object. Fasta files can be .gz. Sequences are \
        reverse complemented by default.'
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
    parser.add_argument('--split_train', type=float, default=0.90,
                        help='proportion of training data (DEFAULT: 0.90)')
    parser.add_argument('--split_test', type=float, default=0.05,
                        help='proportion of testing data (DEFAULT: 0.05)')
    parser.add_argument('--split_val', type=float, default=0.05,
                        help='proportion of validation data (DEFAULT: 0.05)')
    parser.add_argument('--no_reverse_complement', action="store_false",
                        help='turn off reverse complement (DEFAULT: ON)')

    args = parser.parse_args()
    infile_path = args.infile_path
    control_dist = args.control_dist
    tokeniser_path = args.tokeniser_path
    outfile_dir = args.outfile_dir
    special_tokens = args.special_tokens
    split_train = args.split_train
    split_test = args.split_test
    split_val = args.split_val
    do_reverse_complement = args.no_reverse_complement

    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")

    if not os.path.isdir(outfile_dir):
        os.makedirs(outfile_dir)

    warn("Any commas in fasta headers will be replaced with __!")

    # TODO: theres probably a better way to optimise this, all disk operations
    #   write seqs as sql db and read sequentially into pd df?
    #   but need to handle the conversion to huggingface dataset object also
    tmp_control = "".join([outfile_dir, "/.null.tmp"])
    if os.path.exists(tmp_control):
        os.remove(tmp_control)
    with open(tmp_control, mode="a+") as tmp:
        with screed.open(control_dist) as nullfile:
            for read in nullfile:
                head = read.name.replace(",", "__")
                seq = read.sequence.upper()
                tmp.write(head + "," + seq + "\n")
                if do_reverse_complement is True:
                    tmp.write("__".join(["RC", head]) + ",")
                    tmp.write(reverse_complement(seq) + "\n")

    tmp_infile = "".join([outfile_dir, "/.data.tmp"])
    if os.path.exists(tmp_infile):
        os.remove(tmp_infile)
    with open(tmp_infile, mode="a+") as tmp:
        with screed.open(infile_path) as seqfile:
            for read in seqfile:
                head = read.name.replace(",", "__")
                seq = read.sequence.upper()
                tmp.write(head + "," + seq + "\n")
                if do_reverse_complement is True:
                    tmp.write("__".join(["RC", head]) + ",")
                    tmp.write(reverse_complement(seq) + "\n")

    tmp_hf_out = "".join([outfile_dir, "data.hf.csv"])
    if os.path.exists(tmp_hf_out):
        os.remove(tmp_hf_out)
    with open(tmp_hf_out, mode="a+") as tmp_out:
        tmp_out.write("idx,feature,labels\n")
        seqs = pd.read_csv(tmp_control, chunksize=10000, sep=",", header=None)
        for i in seqs:
            i.rename(columns={0: "idx", 1: "feature"}, inplace=True)
            i["labels"] = 0
            tmp_out.write(i.to_csv(index=False, header=False, sep=","))
        os.remove(tmp_control)
        seqs = pd.read_csv(tmp_infile, chunksize=10000, sep=",", header=None)
        for i in seqs:
            i.rename(columns={0: "idx", 1: "feature"}, inplace=True)
            i["labels"] = 1
            tmp_out.write(i.to_csv(index=False, header=False, sep=","))
        os.remove(tmp_infile)

    # configure data into a huggingface compatible dataset object
    # see https://huggingface.co/docs/datasets/access
    data = load_dataset('csv', data_files=tmp_hf_out, split="train")
    os.remove(tmp_hf_out)

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

    print("\nDATASET BEFORE SPLIT:\n", dataset)
    dataset = split_datasets(
        dataset, outfile_dir, train=split_train, test=split_test, val=split_val
        )
    print("\nDATASET AFTER SPLIT:\n", dataset)

    print("\nSAMPLE DATASET ENTRY:\n", dataset["train"][0], "\n")

    print("SAMPLE TOKEN MAPPING FOR FIRST 5 TOKENS IN SEQ:",)
    for i in dataset["train"][0]["input_ids"][0:5]:
        print("TOKEN ID:", i, "\t|", "TOKEN:", tokeniser.decode(i))

    col_torch = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    dataset.set_format(type='torch', columns=col_torch)
    dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=1)
    print("\nSAMPLE PYTORCH FORMATTED ENTRY:\n", next(iter(dataloader)))

if __name__ == "__main__":
    main()