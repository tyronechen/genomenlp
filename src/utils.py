#!/usr/bin/python
# generic tools
import itertools
from math import ceil
from random import choices, shuffle

def bootstrap_seq(seq: str, block_size: int=2):
    """Take a string and shuffle it in blocks of N length"""
    chunks, chunk_size = len(seq), block_size
    seq = [ seq[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    shuffle(seq)
    return "".join(seq)

def generate_from_freq(seq: str, block_size: int=2,
                       alphabet: list=["A","C","G","T"], offset: float=0.01):
    """Take a string and sample from freq distribution to fill up seq length"""
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
    """Take a dna string as input and return reverse complement"""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return "".join([complement[base] for base in dna[::-1]])
