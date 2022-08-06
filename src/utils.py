#!/usr/bin/python
# generic tools
import itertools
import os
from math import ceil
from random import choices, shuffle
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

        Input: 'ACGT'

        Output: 'GTAC'

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
        n-length blocks (eg: [AA, AC, ..] or [AAA, AAC, ...]).

        Input: 'AAAACGT'

        Output: 'ACGTAAA'

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

        Input: 'ACGT'

        Output: 'TGCA'

        Note that no sequence cleaning is performed, 'N' gets mapped to itself.
    """
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return "".join([complement[base] for base in dna[::-1]])

def get_tokens_from_sp(tokeniser_path: str,
                       special_tokens: list=["<s>", "</s>", "<unk>", "<pad>",
                       "<mask>"]):
    """Take path to SentencePiece tokeniser + special tokens, return tokens

    The input `tokeniser_path` is a `json` file generated from the HuggingFace
    implementation of SentencePiece.

    Args:
        tokeniser_path (str): Path to sequence tokens file (from SentencePiece)
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

    The input `tokeniser_path` is a `json` file generated from the HuggingFace
    implementation of SentencePiece.

    Args:
        tokeniser_path (str): Path to sequence tokens file (from SentencePiece)
        special_tokens (list[str]): Special tokens to substitute for
        outfile_dir (str): Path to output plots

    Returns:
        matplotlib.pyplot:

        Token histogram plots are written to `outfile_dir` in `png` and `pdf`
        formats.
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
