#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# tokenise fasta sequences with sentencepiece and export as json file
import argparse
import gzip
import os
import sys
from warnings import warn
from datasets import load_dataset
import pandas as pd
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast
from utils import remove_stopwords, _cite_me

def main():
    parser = argparse.ArgumentParser(
        description='Take gzip csv file(s), run SentencePiece and export json.'
    )
    parser.add_argument('-i', '--infile_paths', type=str, default=None, nargs="+",
                        help='path to files with English language split by line')
    parser.add_argument('-t', '--tokeniser_path', type=str, default="",
                        help='path to tokeniser.json file to save or load data')
    parser.add_argument('-s', '--special_tokens', type=str, nargs="+",
                        default=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                        help='assign special tokens, eg space and pad tokens \
                        (DEFAULT: ["<s>", "</s>", "<unk>", "<pad>", "<mask>"])')
    parser.add_argument('-c', '--col_name', type=str, default=None,
                        help='name of column to parse data on (DEFAULT: None) \
                        Note that this only works in highmem mode (DEFAULT)')
    parser.add_argument('-e', '--example_seq', type=str, default="hello",
                        help='show token to seq map for a sequence \
                        (DEFAULT: hello)')
    parser.add_argument('--dont_remove_stopwords_en', action="store_false",
                        help='dont remove english language stopwords')
    parser.add_argument('--use_lowmem', action="store_false",
                        help='stream file instead of reading with pandas \
                        (useful when memory is low or file is too big)')

    args = parser.parse_args()
    infile_paths = args.infile_paths
    tokeniser_path = args.tokeniser_path
    special_tokens = args.special_tokens
    example_seq = args.example_seq
    remove_stopwords_en = args.dont_remove_stopwords_en
    use_highmem = args.use_lowmem
    col_name = args.col_name

    if infile_paths == None and tokeniser_path == "":
        raise OSError("Provide either input csv/gz file or existing tokeniser!")

    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")

    # if you want to use sentencepiece directly, here are similar commands:
    # NOTE: transformers uses a slightly different implementation though!
    # spm_train \
    #   --vocab_size=2000 \
    #   --input=infile.fa \
    #   --model_prefix=tmp_model \
    #   --normalization_rule_name=identity \
    #   --model_type=unigram \
    #   --max_sentence_length=2048
    # spm_export_vocab --model=tmp_model.model --output=out.txt

    if remove_stopwords_en is True:
        infile_paths = [
            remove_stopwords(dataset=i, column=col_name, highmem=use_highmem)
            for i in infile_paths
            ]
        print("\nOUTPUT FILES WITHOUT STOPWORDS:\n", infile_paths, "\n")

    dataset = load_dataset("csv", data_files=infile_paths)

    if infile_paths:
        tokeniser = SentencePieceUnigramTokenizer()
        tokeniser.train_from_iterator(
            dataset["train"]["text"],
            unk_token="<unk>",
            vocab_size=32000,
            show_progress=True,
            special_tokens=special_tokens,
            # limit_alphabet=500,
            # min_frequency=5,
        )
        if tokeniser_path != "":
            if os.path.exists(tokeniser_path):
                warn("This will overwrite any existing tokeniser!")
            tokeniser.save(tokeniser_path)

    if os.path.exists(tokeniser_path):
        tokeniser = PreTrainedTokenizerFast(tokenizer_file=tokeniser_path)

    if example_seq:
        print("Sample input sequence:", example_seq)
        model_inputs = tokeniser(example_seq)

        tokens = tokeniser.tokenize(example_seq)
        ids = tokeniser.convert_tokens_to_ids(tokens)
        print("Sample tokenised:", ids)

        for i in ids:
            print("Token::k-mer map:", i, "\t::", tokeniser.decode(i))

if __name__ == "__main__":
    main()
    _cite_me()
