#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# create huggingface dataset given input fasta, control and tokeniser files
import argparse
import os
import sys
from warnings import warn
from datasets import ClassLabel
import torch
from datasets import load_dataset
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast
from utils import chunk_text, split_datasets, _cite_me

def main():
    parser = argparse.ArgumentParser(
        description='Take control and test csv files, tokeniser and convert \
        to HuggingFace🤗 dataset object. csv files can be .gz.'
    )
    parser.add_argument('infile_path', type=str, help='path to csv/gz file')
    parser.add_argument('tokeniser_path', type=str, help='load tokeniser file')
    parser.add_argument('title', type=str, help='name of the column in the \
                        csv file which contains a unique identifier')
    parser.add_argument('labels', type=str, help='name of the column in the \
                        csv file which contains labels')
    parser.add_argument('content', type=str, help='name of the column in the \
                        csv file which contains content')
    parser.add_argument('-d', '--control_dist', type=str, default=None,
                        help='supply category 2')
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
    parser.add_argument('--no_shuffle', action="store_false",
                        help='turn off shuffle for data split (DEFAULT: ON)')

    args = parser.parse_args()
    infile_path = args.infile_path
    control_dist = args.control_dist
    tokeniser_path = args.tokeniser_path
    outfile_dir = args.outfile_dir
    special_tokens = args.special_tokens
    chunk = args.chunk
    split_train = args.split_train
    split_test = args.split_test
    split_val = args.split_val
    labels = args.labels
    content = args.content
    title = args.title
    do_shuffle = args.no_shuffle

    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")

    if not os.path.isdir(outfile_dir):
        os.makedirs(outfile_dir)

    # chunk text into n-length blocks
    if chunk != None:
        print("DIVIDE TEXT INTO BLOCKS OF LENGTH:", chunk, "\n")
        tmp_infile = "".join([outfile_dir, "/data.tmp"])
        if os.path.exists(tmp_infile):
            os.remove(tmp_infile)
        chunk_text(infile_path, tmp_infile, title, labels, content, chunk)
        data = load_dataset('csv', data_files=tmp_infile, split="train")
        if os.path.exists(tmp_infile):
            os.remove(tmp_infile)
    else:
        data = load_dataset('csv', data_files=infile_path, split="train")

    # configure data into a huggingface compatible dataset object
    # see https://huggingface.co/docs/datasets/access

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
    dataset = data.map(lambda data: tokeniser(data[content]), batched=True)
    # https://discuss.huggingface.co/t/class-labels-for-custom-datasets/15130
    dataset = dataset.class_encode_column(labels)

    print("\nDATASET FEATURES:\n", dataset.features, "\n")
    print("CLASS NAMES:", dataset.features[labels].names)
    print("CLASS COUNT:", dataset.features[labels].num_classes)
    print("\nDATASET HAS FOLLOWING SPECIFICATIONS:\n", dataset)

    print("\nDATASET BEFORE SPLIT:\n", dataset)
    dataset = split_datasets(
        dataset, outfile_dir, train=split_train, test=split_test, val=split_val,
        shuffle=do_shuffle
        )
    print("\nDATASET AFTER SPLIT:\n", dataset)

    # print("\nSAMPLE DATASET ENTRY:\n", dataset["train"][0], "\n")

    print("SAMPLE TOKEN MAPPING FOR FIRST 5 TOKENS IN SEQ:",)
    for i in dataset["train"][0]["input_ids"][0:5]:
        print("TOKEN ID:", i, "\t|", "TOKEN:", tokeniser.decode(i))

    # col_torch = ['input_ids', 'token_type_ids', 'attention_mask', labels]
    # dataset.set_format(type='torch', columns=col_torch)
    # dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=1)
    # print("\nSAMPLE PYTORCH FORMATTED ENTRY:\n", next(iter(dataloader)))

if __name__ == "__main__":
    main()
    _cite_me()
