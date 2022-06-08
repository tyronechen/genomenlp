#!/usr/bin/python
# tokenise fasta sequences with sentencepiece and export as json file
import argparse
import gzip
import os
from warnings import warn
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast

def _gzip_iterator(infile_paths):
    for path in infile_paths:
        with gzip.open(path, "rt") as f:
            for line in f:
                if not line.startswith(">"):
                    yield line.strip().upper()

def main():
    parser = argparse.ArgumentParser(
        description='Take gzip fasta file(s), run SentencePiece and export json.'
    )
    parser.add_argument('-i', '--infile_paths', type=str, default=None, nargs="+",
                        help='path to files with biological seqs split by line')
    parser.add_argument('-t', '--tokeniser_path', type=str, default="",
                        help='path to tokeniser.json file to save or load data')
    parser.add_argument('-s', '--special_tokens', type=str, nargs="+",
                        default=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                        help='assign special tokens, eg space and pad tokens \
                        (DEFAULT: ["<s>", "</s>", "<unk>", "<pad>", "<mask>"])')
    parser.add_argument('-e', '--example_seq', type=str, default="AACCGGTT",
                        help='show token to seq map for a sequence \
                        (DEFAULT: AACCGGTT)')
    args = parser.parse_args()
    infile_paths = args.infile_paths
    tokeniser_path = args.tokeniser_path
    special_tokens = args.special_tokens
    example_seq = args.example_seq

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

    if infile_paths:
        tokeniser = SentencePieceUnigramTokenizer()
        tokeniser.train_from_iterator(
            _gzip_iterator(infile_paths),
            unk_token="<unk>",
            vocab_size=2000,
            show_progress=True,
            special_tokens=special_tokens,
            # limit_alphabet=500,
            # min_frequency=5,
        )
        if tokeniser_path != "":
            if os.path.exists(tokeniser_path):
                warn("This will overwrite existing tokeniser!")
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
