#!/usr/bin/python
# generic tools
import itertools
import os
from math import ceil
from random import choices, shuffle
from warnings import warn
import pandas as pd
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast

def bootstrap_seq(seq: str, block_size: int=2):
    """Take a string and reshuffle it in blocks of N length.

    Shuffles a sequence in the user-defined block size. Joins the
    sequence back together at the end.

    Compare :py:func:`generate_from_freq`.

    Args:
        seq (str): A string of biological sequence data.
        block_size (int): An integer specifying the size of block to shuffle.

    Returns:
        str:

        A reshuffled string of the same length as the original input

        Input: ``ACGT``

        Output: ``GTAC``

        If the reconstructed seq exceeds seq length it will be truncated.
    """
    chunks, chunk_size = len(seq), block_size
    seq = [ seq[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    shuffle(seq)
    return "".join(seq)

def generate_from_freq(seq: str, block_size: int=2,
                       alphabet: list=["A","C","G","T"], offset: float=0.01):
    """Take a string and sample from freq distribution to fill up seq length.

    Compare :py:func:`bootstrap_seq`.

    Args:
        seq (str): A string of biological sequence data
        block_size (int): Size of block to shuffle
        alphabet (list[str]): Biological alphabet present in input sequences
        offset (float): Adding offset avoids 0 division errors in small datasets

    Returns:
        str:

        Resampled sequence with matching frequency distribution of the same
        length as the original input. Frequency distribution is sampled as
        n-length blocks (eg: ``[AA, AC, ..]`` or ``[AAA, AAC, ...]``).

        Input: ``AAAACGT``

        Output: ``ACGTAAA``

        If the reconstructed seq exceeds seq length it will be truncated.
    """
    if len(seq) == 0:
        return
    assert block_size <= len(seq), "Block size exceeds sequence length!"
    to_count = ["".join(i) for i in itertools.product(alphabet, repeat=block_size)]
    count = [str.count(seq, x) + offset for x in to_count]
    freq = dict(zip(to_count, [x / len(count) for x in count]))
    draw_size = ceil(len(seq) / block_size)
    new = choices(list(freq.keys()), weights=freq.values(), k=draw_size)
    return "".join(new)[:len(seq)]

def reverse_complement(dna: str):
    """Take a dna string as input and return reverse complement.

    Args:
        dna (str): A string of dna sequence data.

    Returns:
        str:

        Reverse complemented DNA string.

        Input: ``ACGT``

        Output: ``TGCA``

        Note that no sequence cleaning is performed, 'N' gets mapped to itself.
        Uppercase is assumed. Does not work on RNA!
    """
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return "".join([complement[base] for base in dna[::-1]])

def get_tokens_from_sp(tokeniser_path: str,
                       special_tokens: list=["<s>", "</s>", "<unk>", "<pad>",
                       "<mask>"]):
    """Take path to ``SentencePiece`` tokeniser + special tokens, return tokens

    The input ``tokeniser_path`` is a ``json`` file generated from the
    ``HuggingFace`` implementation of ``SentencePiece``.

    Args:
        tokeniser_path (str): Path to sequence tokens file (from ``SentencePiece``)
        special_tokens (list[str]): Special tokens to substitute for

    Returns:
        list:

        A list of cleaned tokens corresponding to variable length k-mers.
    """
    # if we dont specify the special tokens below it will break
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
    return [x.replace("▁", "") for x in list(tokeniser.vocab.keys())]

def plot_token_dist(tokeniser_path: str, special_tokens: list=["<s>", "</s>",
                    "<unk>", "<pad>", "<mask>"], outfile_dir: str="./"):
    """Plot distribution of token lengths. Calls :py:func:`get_tokens_from_sp`

    The input ``tokeniser_path`` is a ``json`` file generated from the
    ``HuggingFace`` implementation of ``SentencePiece``.

    Args:
        tokeniser_path (str): Path to sequence tokens file (from ``SentencePiece``)
        special_tokens (list[str]): Special tokens to substitute for
        outfile_dir (str): Path to output plots

    Returns:
        matplotlib.pyplot:

        Token histogram plots are written to ``outfile_dir`` in ``png`` and
        ``pdf`` formats.
    """
    tokens = get_tokens_from_sp(tokeniser_path, special_tokens)
    for special_token in special_tokens:
        tokens.remove(special_token)
    tokens_len = [len(x) for x in tokens]

    for_plot = pd.DataFrame(pd.Series(tokens_len))
    for_plot.columns = ["Selected k-mer lengths (base pairs)"]
    for_plot.index.name = "Quantity (units)"

    hist = for_plot.plot(kind="hist", grid=False, legend=False)
    hist.set_xlabel("Selected k-mer lengths (base pairs)")
    title = "".join(
        ["Selected k-mer length distribution (of ", str(len(tokens_len)), ")"]
        )
    hist.set_title(title)
    plt_out = ["".join(
        [outfile_dir, "kmer_length_histogram.", i]
        ) for i in ["pdf", "png"]]
    [plt.savefig(i, dpi=300) for i in plt_out]
    return hist

def remove_stopwords(dataset: str, column: str=None, highmem: bool=True):
    """Remove English language stopwords from text. Stopwords are obtained from
    ``SpaCy 3.2.4``.

    Args:
        dataset (str): A path to a comma separated ``.csv`` file
        column (str): The name of the column to be cleaned. If no column text is
            provided (*default*), parses all columns. This option is disabled if
            highmem is set to ``False``!
        highmem (bool): If ``True`` (*default*), uses ``pandas`` to operate on
            the file. If ``False``, parses the file line by line, overriding
            column selection!

    Returns:
        str:

        New file path with removed stopwords, named ``dataset.CLEAN``.
        Note that stopwords with leading uppercase are also removed.
        For example "the" and "The" will be treated the same and removed.
        To obtain the stopwords list for English used in this function::

            #!/bin/bash
            python -m spacy download en

            #!/usr/bin/python
            import spacy
            sp = spacy.load('en_core_web_sm')
            stopwords_en = sp.Defaults.stop_words
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
