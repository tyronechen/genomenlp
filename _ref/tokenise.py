#!/usr/bin/python
# tokenise fasta sequences with sentencepiece
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
        description='Take fasta-like files, run SentencePiece Tokeniser.'
    )
    parser.add_argument('-i', '--infile_paths', type=str, default=None, nargs="+",
                        help='path to files with biological seqs split by line')
    parser.add_argument('-t', '--tokeniser_path', type=str, default="",
                        help='path to tokeniser.json file to save or load data')
    args = parser.parse_args()
    infile_paths = args.infile_paths
    tokeniser_path = args.tokeniser_path

    # files = ["foo.fa.gz"]
    # spm_train --vocab_size=2000 --input=foo.fa --model_prefix=tmp_model --normalization_rule_name=identity --model_type=unigram --max_sentence_length=2048
    # spm_export_vocab --model=tmp_model.model --output=out.txt

    if infile_paths:
        tokeniser = SentencePieceUnigramTokenizer()
        tokeniser.train_from_iterator(
            _gzip_iterator(infile_paths),
            unk_token="<unk>",
            vocab_size=2000,
            show_progress=True,
            # limit_alphabet=500,
            # min_frequency=5,
            # special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
        )
        # tokeniser = PreTrainedTokenizerFast(
        #     tokenizer_object=tokeniser
        # )
        if tokeniser_path != "":
            if os.path.exists(tokeniser_path):
                warn("This will overwrite existing tokeniser!")
            tokeniser.save(tokeniser_path)

    if os.path.exists(tokeniser_path):
        tokeniser = PreTrainedTokenizerFast(tokenizer_file=tokeniser_path)

    sequence = "GGCCCGTCCGCGCCAG"
    print("Sample input sequence:", sequence)
    model_inputs = tokeniser(sequence)

    tokens = tokeniser.tokenize(sequence)
    ids = tokeniser.convert_tokens_to_ids(tokens)
    print("Sample tokenised:", ids)

    # this will give you same result as above block
    # output = tokeniser.encode("GGCCCGTCCGCGCCAG")

    for i in [1, 55, 155, 53, 558]:
        print("Token::k-mer map:", i, "::", tokeniser.decode(i))

if __name__ == "__main__":
    main()
