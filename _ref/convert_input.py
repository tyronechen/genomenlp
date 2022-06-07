#!/usr/bin/python
# take a bam/sam/fasta/q file, convert to custom formatted bed-like file
import argparse
import math
import os
import random
from warnings import warn
import pandas as pd
import pysam
import screed
from tqdm import tqdm

class MakeCustomBed(object):
    """
    Check file type, parse input, generate custom bed-like file for lime.

    Arguments:
        infile - input file
        filetype - input file type, must be one of [bam, fasta, fastq, sam]
        subsample - number of lines to take from file (DEFAULT: 100)
            if the number of resampled reads are too low AND the file is sorted,
            the subsampling will not be evenly distributed across the file.
        sorted - is the file sorted, assume True by default (DEFAULT: True)
            if sorted, will draw seqs at random from file (slow).
            if unsorted, will take from top of file directly (a lot faster).
            you can unsort file with other methods if you want before passing in
        outfile - write output file to here (DEFAULT: None)
        threads - number of cpus to use (DEFAULT: 2)
            this is effective for bam and sam files, not implemented for fasta/q
        gzip - gzip output file by default (DEFAULT: False)
            this creates some overhead as the file is written to line by line.
    """
    def __init__(self, infile: str, filetype: str=None, subsample: int=100,
                 sorted: bool=True, outfile: str=None, threads: int=2,
                 gzip: bool=False):
        super(MakeCustomBed, self).__init__()
        self.infile = infile
        if filetype is None:
            self.filetype = self.infile.split(".")[-1]
            filetype = " ".join([
                "No filetype provided, inferring from file name:", self.filetype
                ])
            warn(filetype)
        else:
            self.filetype = filetype
        self.sorted = sorted
        self.threads = threads
        self.filetypes = ["bam", "fasta", "fastq", "sam"]
        self.subsample = subsample
        self.outfile = outfile
        self.prob = None
        self.total = None
        if self.subsample < 10000 and sorted is True:
            warn("Resampling may be ineffective, increase subsample to >=10000")
        if gzip is True:
            if not self.outfile.split(".")[-1] == "gz":
                self.outfile = ".".join([self.outfile, "gz"])
        if outfile is None:
            self.outfile = ".".join([self.infile, "bed", "gz"])
        self.outfile_exists()

    def outfile_exists(self):
        """Check if output file exists and remove if found."""
        if os.path.exists(self.outfile):
            warn(" ".join([self.outfile, "exists, overwriting!"]))
            os.remove(self.outfile)

    def decide(self, probability):
        """Return True if random float [0,1)."""
        return random.random() < probability

    def define_probability(self):
        """Round up slightly to make sure the sequence quota is hit."""
        base_prob = self.subsample / self.total
        significant = f'{float(f"{base_prob:.2g}"):g}'
        increment_pos = len(significant.split(".")[-1])
        return float(significant) + (10**-increment_pos)

    def parse_file(self):
        """Check filetype for parsing and return generator object."""
        if self.filetype not in self.filetypes:
            raise RuntimeError(
                " ".join(["File must be a:", " ".join(self.filetypes)])
                )
        save = pysam.set_verbosity(0)

        if self.filetype == "bam":
            if self.sorted is True:
                self.total = int(pysam.view("-c", self.infile).rstrip())
                self.prob = self.define_probability()
            data = self.parse_bam()

        if self.filetype == "sam":
            if self.sorted is True:
                self.total = int(pysam.view("-c", self.infile).rstrip())
                self.prob = self.define_probability()
            data = self.parse_sam()

        if self.filetype == "fasta":
            if self.sorted is True:
                with screed.open(self.infile) as infile:
                    self.total = int(len([x for x in infile]))
                self.prob = self.define_probability()
            data = self.parse_fasta()

        if self.filetype == "fastq":
            if self.sorted is True:
                with screed.open(self.infile) as infile:
                    self.total = int(len([x for x in infile]))
                self.prob = self.define_probability()
            data = self.parse_fastq()

        pysam.set_verbosity(save)

    def parse_bam(self):
        """Parse a bam file, assign lime fields, write bed-like file."""
        data = pysam.AlignmentFile(self.infile, mode="rb", threads=self.threads)
        if self.sorted is True:
            counter = 0
            for x in tqdm(data.head(self.total), total=self.total):
                if self.decide(self.prob) is True:
                    if x.is_reverse is True:
                        strand = "reverse"
                    else:
                        strand = "forward"
                    if x.is_unmapped is True:
                        mapping = "unmapped"
                        ref_name = "UNMAPPED"
                    else:
                        mapping = "mapped"
                        ref_name = x.reference_name
                    if len(x.positions) != 0:
                        pos_start = x.positions[0]
                        pos_end = x.positions[-1]
                    else:
                        pos_start = 0
                        pos_end = 0
                    out = [ref_name, x.query_name, x.query_name, strand,
                           pos_start, pos_end, x.seq, mapping]
                    out = pd.DataFrame(out).T
                    out.to_csv(
                        self.outfile,mode='a',header=False,index=False,sep="\t"
                        )
                    counter += 1
                    if counter == self.subsample:
                        break
                else:
                    continue
        else:
            for x in tqdm(data.head(self.subsample), total=self.subsample):
                if x.is_reverse is True:
                    strand = "reverse"
                else:
                    strand = "forward"
                if x.is_unmapped is True:
                    mapping = "unmapped"
                    ref_name = "UNMAPPED"
                else:
                    mapping = "mapped"
                    ref_name = x.reference_name
                if len(x.positions) != 0:
                    pos_start = x.positions[0]
                    pos_end = x.positions[-1]
                else:
                    pos_start = 0
                    pos_end = 0
                out = [ref_name, x.query_name, x.query_name, strand,
                    pos_start, pos_end, x.seq, mapping]
                out = pd.DataFrame(out).T
                out.to_csv(
                    self.outfile,mode='a',header=False,index=False,sep="\t"
                    )

    def parse_fasta(self):
        """Parse a fasta file, assign lime fields, write bed-like file."""
        if self.sorted is True:
            counter = 0
            with open(self.outfile, mode="a+") as outfile:
                with screed.open(self.infile) as seqfile:
                    counter = 0
                    for read in tqdm(seqfile, total=self.total):
                        if self.decide(self.prob) is True:
                            name = read.name
                            chr_coords = name.split(":")
                            chr = chr_coords[0]
                            start_end = chr_coords[1].split("(")[0].split("-")
                            start = start_end[0]
                            end = start_end[1]
                            seq = read.sequence
                            outfile.write("\t".join(
                                [chr, name, name, "forward", str(start),
                                 str(end), seq, "NA", "\n"]
                                ))
                            counter += 1
                            if counter == self.subsample:
                                break
                        else:
                            continue
        else:
            with open(self.outfile, mode="a+") as outfile:
                with screed.open(self.infile) as seqfile:
                    counter = 0
                    for read in tqdm(seqfile, total=self.subsample):
                        name = read.name
                        chr_coords = name.split(":")
                        chr = chr_coords[0]
                        start_end = chr_coords[1].split("(")[0].split("-")
                        start = start_end[0]
                        end = start_end[1]
                        seq = read.sequence
                        outfile.write("\t".join(
                            [chr, name, name, "forward", str(start),
                             str(end), seq, "NA", "\n"]
                            ))
                        counter += 1
                        if counter == self.subsample:
                            break

    def parse_fastq():
        raise NotImplementedError("Not yet implemented")

    def parse_sam():
        """Parse a sam file, assign lime fields, write bed-like file."""
        data = pysam.AlignmentFile(self.infile, mode="r", threads=self.threads)
        if self.sorted is True:
            counter = 0
            for x in tqdm(data.head(self.total), total=self.total):
                if self.decide(self.prob) is True:
                    if x.is_reverse is True:
                        strand = "reverse"
                    else:
                        strand = "forward"
                    if x.is_unmapped is True:
                        mapping = "unmapped"
                        ref_name = "UNMAPPED"
                    else:
                        mapping = "mapped"
                        ref_name = x.reference_name
                    if len(x.positions) != 0:
                        pos_start = x.positions[0]
                        pos_end = x.positions[-1]
                    else:
                        pos_start = 0
                        pos_end = 0
                    out = [ref_name, x.query_name, x.query_name, strand,
                           pos_start, pos_end, x.seq, mapping]
                    out = pd.DataFrame(out).T
                    out.to_csv(
                        self.outfile,mode='a',header=False,index=False,sep="\t"
                        )
                    counter += 1
                    if counter == self.subsample:
                        break
                else:
                    continue
        else:
            for x in tqdm(data.head(self.subsample), total=self.subsample):
                if x.is_reverse is True:
                    strand = "reverse"
                else:
                    strand = "forward"
                if x.is_unmapped is True:
                    mapping = "unmapped"
                    ref_name = "UNMAPPED"
                else:
                    mapping = "mapped"
                    ref_name = x.reference_name
                if len(x.positions) != 0:
                    pos_start = x.positions[0]
                    pos_end = x.positions[-1]
                else:
                    pos_start = 0
                    pos_end = 0
                out = [ref_name, x.query_name, x.query_name, strand,
                    pos_start, pos_end, x.seq, mapping]
                out = pd.DataFrame(out).T
                out.to_csv(
                    self.outfile,mode='a',header=False,index=False,sep="\t"
                    )

def main():
    parser = argparse.ArgumentParser(
        description='Convert a bam/sam/fasta/q file to custom formatted bedfile'
    )
    parser.add_argument('infile_path', type=str, help='path to [b/sam/fasta/q]')
    parser.add_argument('-i', '--is_sorted', action="store_false",
                        help='is the input file sorted? assume (DEFAULT: True)')
    parser.add_argument('-o', '--outfile_path', type=str, default=None,
                        help='write bed-like file to this path (DEFAULT: None)')
    parser.add_argument('-s', '--subsample', type=int, default=100,
                        help='number of sequences to subsample (DEFAULT: 100)')
    parser.add_argument('-t', '--threads', type=int, default=2,
                        help='number of cpus to use (DEFAULT: 2)')

    args = parser.parse_args()
    infile = args.infile_path
    outfile = args.outfile_path
    is_sorted = args.is_sorted
    subsample = args.subsample
    threads = args.threads

    data = MakeCustomBed(
        infile=infile,
        outfile=outfile,
        subsample=subsample,
        threads=threads,
        sorted=is_sorted
        )
    data.parse_file()

if __name__ == '__main__':
    main()
