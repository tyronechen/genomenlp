# README

This explains the use of `tokenise.py`. We run a variant of SentencePiece on biological sequence data.

## Requirements

All available via `pip install`:

```
python==3.10.4
transformers==4.18.0
tokenizers==0.11.6
```

## Source data

Any fasta file can be used. However, for reproducibility the data in this example can be obtained by running `../data/get_data.sh`.

## Results

Running the code as below:

```
python tokenise.py -i path/to/infile.fa -t path/to/tokeniser.json
```

You will obtain a `json` file with weights for each token. Any special tokens you add will also be present. This will be used in the next step of creating a `huggingface` compatible dataset object.

## Notes

[The original SentencePiece github repository is available here.](https://github.com/google/sentencepiece) Please note that we use the variant of SentencePiece in the [`tokenizers` package specifically, which is also available on github.](https://github.com/huggingface/tokenizers)

## Usage

```
python tokenise.py -h
usage: tokenise.py [-h] [-i INFILE_PATHS [INFILE_PATHS ...]] [-t TOKENISER_PATH]
                   [-s SPECIAL_TOKENS [SPECIAL_TOKENS ...]] [-e EXAMPLE_SEQ]

Take gzip fasta file(s), run SentencePiece and export json.

options:
  -h, --help            show this help message and exit
  -i INFILE_PATHS [INFILE_PATHS ...], --infile_paths INFILE_PATHS [INFILE_PATHS ...]
                        path to files with biological seqs split by line
  -t TOKENISER_PATH, --tokeniser_path TOKENISER_PATH
                        path to tokeniser.json file to save or load data
  -s SPECIAL_TOKENS [SPECIAL_TOKENS ...], --special_tokens SPECIAL_TOKENS [SPECIAL_TOKENS ...]
                        special tokens assigned
  -e EXAMPLE_SEQ, --example_seq EXAMPLE_SEQ
                        show token to seq map for an example sequence
```
