#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# generate synthetic sequences given fasta file
import argparse
import gzip
import os
import sys
from warnings import warn
import pandas as pd
import screed
from tqdm import tqdm
from utils import bootstrap_seq, generate_from_freq, reverse_complement, _cite_me

def main():
    parser = argparse.ArgumentParser(
        description='Take fasta files, generate synthetic sequences. \
        Can accept .gz files.'
    )
    parser.add_argument('infile_path', type=str, help='path to fasta/gz file')
    parser.add_argument('-b', '--block_size', type=int, default=2,
                        help='size of block to generate synthetic sequences \
                        from as negative control (DEFAULT: 2)')
    parser.add_argument('-c', '--control_dist', type=str, default="frequency",
                        help='generate control distribution by [ bootstrap | \
                        frequency ] (DEFAULT: frequency)')
    parser.add_argument('-o', '--outfile', type=str, default="out.fa",
                        help='write synthetic sequences (DEFAULT: "out.fa")')
    parser.add_argument('--do_reverse_complement', action="store_true",
                        help='turn on reverse complement (DEFAULT: OFF)')

    args = parser.parse_args()
    infile_path = args.infile_path
    block_size = args.block_size
    control_dist = args.control_dist
    outfile = args.outfile
    do_reverse_complement = args.do_reverse_complement

    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")

    if not control_dist in ["bootstrap", "frequency"]:
        raise OSError("Make null distribution by [ bootstrap | frequency] only!")

    if os.path.isfile(outfile):
        raise OSError("Existing output with same name detected! Rename or remove")

    # only to get count for tqdm
    with screed.open(infile_path) as seqfile:
        count = 0
        for read in seqfile:
            count += 1

    # do reverse complement and negative control generation in the same loop
    with open(outfile, mode="a+") as outfile:
        with screed.open(infile_path) as seqfile:
            for read in tqdm(seqfile, total=count):
                head = read.name
                seq = read.sequence.upper()

                if do_reverse_complement is True:
                    rc_head = "".join([">RC_", head, "\n"])
                    rc_seq = reverse_complement(seq) + "\n"
                    outfile.write(rc_head)
                    outfile.write(rc_seq)

                if control_dist == "bootstrap":
                    null_head = "".join([">NULL_B_", head, "\n"])
                    null_seq = bootstrap_seq(seq, block_size) + "\n"
                    outfile.write(null_head)
                    outfile.write(null_seq)
                    if do_reverse_complement is True:
                        null_head_rc = "".join([">NULL_B_RC_", head, "\n"])
                        null_seq_rc = bootstrap_seq(rc_seq, block_size)

                if control_dist == "frequency":
                    null_head = "".join([">NULL_F_", head, "\n"])
                    null_seq = generate_from_freq(seq, block_size) + "\n"
                    outfile.write(null_head)
                    outfile.write(null_seq)
                    if do_reverse_complement is True:
                        null_head_rc = "".join([">NULL_F_RC_", head, "\n"])
                        null_seq_rc = generate_from_freq(rc_seq, block_size)

if __name__ == "__main__":
    main()
    _cite_me()
