#!/usr/bin/python
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

def remove_stopwords(dataset: str, column: str=None, highmem: bool=True):
    """Remove english language stopwords from text. Stopwords from SpaCy 3.2.4.

    Args:
        dataset (str): A path to a comma separated .csv file
        column (str): The name of the column to be cleaned
        highmem (bool): Use pandas by default, if not stream file through

    Returns:
        str:

        New file path with removed stopwords, named ``dataset.CLEAN``.

        ``
        # to obtain stopwords list

        #!/bin/bash
        python -m spacy download en

        #!/usr/bin/python
        import spacy
        sp = spacy.load('en_core_web_sm')
        stopwords_en = sp.Defaults.stop_words
        ``
    """

    # obtained from spacy
    stopwords_en = {
        'twelve', 'along', 'for', 'most', '‘d', 'as', 'the', 'in', 'ever',
        'themselves', 'whole', 'here', 'do', 'so', 'elsewhere', 'therefore',
        "'ve", '‘re', 'alone', 'make', 'just', '’ve', 'on', 'eight', 'such',
        'hereupon', "'re", 'whereas', 'is', 'might', 'thereupon', 'yours',
        'because', 'almost', 'how', 'amongst', 'it', 'everything', 'while',
        'anyone', 'whom', 'namely', 'hereafter', 'during', 'quite', "n't",
        'those', 'every', 'beforehand', 'wherein', 'his', 'our', 'beyond', 'no',
        'done', 'six', 'used', 'become', 'within', 'seems', 'have', 'well',
        '’s', 'top', 'keep', 'another', 'none', 'although', 'per', '‘s',
        'which', 'toward', 'four', 'first', 'anyway', '’re', 'her', 'take',
        'am', 'himself', 'too', 'call', 'wherever', 'down', 'into', 'up',
        'unless', 'seemed', 'what', 'thru', 'hundred', 'your', "'m", 'each',
        'does', 'though', 'name', 'hers', 'afterwards', 'some', 'front', 'made',
        'show', 'its', 'perhaps', 'were', 'other', 'than', 'without', 'least',
        'enough', 'by', 'until', 'him', 'from', 'amount', 'say', 'became',
        'yourself', 'throughout', 'about', 'where', 'can', 'former', 'two',
        'rather', 'anywhere', 'off', 'indeed', 'give', 'mostly', 'only', 'back',
        'go', 'put', 'more', 'onto', 'somehow', '’d', '’m', 'ca', 'bottom',
        'cannot', '‘ll', 'we', 'any', 'would', 'nor', 'whither', 'one', 'n’t',
        'herself', 'at', 'everywhere', 'few', 'been', 'between', 'please',
        'below', 'around', 'regarding', 'using', 'across', 'several', 'whereby',
        'fifty', 'less', 'someone', 'get', 'before', 'seeming', 'since',
        'therein', 'myself', 'be', 'sometime', 'to', 'was', 'whenever',
        'latterly', 'three', 'nevertheless', 'whereafter', 'still', 'always',
        'five', 'ourselves', 'serious', 'has', 'should', 'their', 'ours',
        'hence', 'empty', 'n‘t', 'upon', 'formerly', 'them', 'itself', 'all',
        'besides', 'i', 'due', 'under', 'others', 'through', 'whose', 'if',
        'did', 'why', 'mine', 'beside', 'third', 'moreover', 'otherwise', 'via',
        'whoever', "'d", 'or', 'together', 'whence', 'doing', 'thence', 'he',
        'they', 'sometimes', "'s", 'see', 'never', 'against', 'over',
        'whatever', 'next', 'yourselves', 'now', 'part', 'even', 'except',
        'twenty', 'once', 'both', 'thereby', 'ten', 'full', 'anyhow', 'also',
        'noone', 'among', 'are', 'very', '‘ve', 'herein', 'eleven', 'and',
        'after', 'often', 'with', 'nowhere', 'may', 'becoming', 'really', '‘m',
        'my', 'whereupon', 'fifteen', 'same', 'various', 'again', 'nine', 'of',
        'you', 'a', 'behind', 'everyone', '’ll', 'side', 'else', 'further',
        'an', 'either', 'last', "'ll", 'could', 'will', 'must', 'who', 'forty',
        'neither', 'when', 'being', 'move', 'she', 'there', 'us', 'nothing',
        'seem', 'had', 'many', 'that', 'becomes', 'not', 'already', 'towards',
        'this', 'but', 'whether', 'sixty', 'thus', 'these', 'then', 'nobody',
        'anything', 'latter', 're', 'much', 'hereby', 'something', 'me', 'yet',
        'thereafter', 'out', 'meanwhile', 'above', 'however', 'somewhere', 'own'
        }
    # we correctly ignore indexes out of range
    stopwords_en_case = {"".join([i[0].upper(), i[1:]]) for i in stopwords_en}
    stopwords_en = stopwords_en.union(stopwords_en_case)

    outfile_path = ".".join([dataset, "CLEAN"])
    if os.path.exists(outfile_path):
        warn("This will overwrite any existing file(s) with the same name!")
        os.remove(outfile_path)

    if highmem is True:
        dataset = pd.read_csv(dataset, sep=",")
        # parse everything by default
        # "的 " is used here as a filler to parse "\nFOO" strings (en) correctly
        if column == None:
            for col in dataset.columns:
                if dataset[col].dtype == "object":
                    dataset[col] = [
                        " ".join(i).replace("的 ", "\n") for i in [
                            [i for i in text.replace("\n", "的 ").split(" ")
                             if not i in stopwords_en]
                                for text in dataset[col]
                            ]
                        ]
        # target a specific column to parse
        else:
            dataset[column] = [
                " ".join(i).replace("的 ", "\n") for i in [
                    [i for i in text.replace("\n", "的 ").split(" ")
                     if not i in stopwords_en]
                        for text in dataset[column]
                    ]
                ]
        dataset.to_csv(outfile_path, index=False)

    else:
        # this hits all columns!
        with open(outfile_path, mode="a+") as outfile:
            with open(dataset) as infile:
                for line in infile:
                    outfile.write(" ".join(
                        [i for i in line.replace("\n", "的 ").split(" ")
                         if not i in stopwords_en]
                        ).replace("的 ", "\n"))
    return outfile_path

def main():
    parser = argparse.ArgumentParser(
        description='Take gzip csv file(s), run SentencePiece and export json.'
    )
    parser.add_argument('-i', '--infile_paths', type=str, default=None, nargs="+",
                        help='path to files with biological seqs split by line')
    parser.add_argument('-t', '--tokeniser_path', type=str, default="",
                        help='path to tokeniser.json file to save or load data')
    parser.add_argument('-s', '--special_tokens', type=str, nargs="+",
                        default=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                        help='assign special tokens, eg space and pad tokens \
                        (DEFAULT: ["<s>", "</s>", "<unk>", "<pad>", "<mask>"])')
    parser.add_argument('-c', '--col_name', type=str, default=None,
                        help='name of column to parse data on (DEFAULT: None) \
                        Note that this only works in highmem mode (DEFAULT)')
    parser.add_argument('-e', '--example_seq', type=str, default="AACCGGTT",
                        help='show token to seq map for a sequence \
                        (DEFAULT: AACCGGTT)')
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
