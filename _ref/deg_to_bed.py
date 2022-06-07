#!/usr/bin/python
# take top N differentially expressed genes and find coordinates for bedtools
import argparse
from gtfparse import read_gtf
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description='Get top genes from DEG table, match to genome coordinates.'
    )
    parser.add_argument('infile_path', type=str, help='path to DEG table')
    parser.add_argument('gtf_file', type=str, help='gtf annotation file')
    parser.add_argument('-c', '--filter_column', type=str, default="FDR",
                        help='column to apply filter value (DEFAULT: FDR)')
    parser.add_argument('-f', '--filter_value', type=float, default=0.05,
                        help='filter out anything below this (DEFAULT: 0.05)')
    parser.add_argument('-g', '--gtf_column', type=str, default="gene_name",
                        help='use this column from gtf (DEFAULT: gene_name)')
    parser.add_argument('-n', '--number_genes', type=int, default=100,
                        help='number of top genes (DEFAULT: 100)')
    parser.add_argument('-o', '--outfile_path', type=str, default=None,
                        help='write bedfile to this path (DEFAULT: None)')
    parser.add_argument('-s', '--start_offset', type=int, default=0,
                        help='extract region N bases upstream (DEFAULT: 0)')

    args = parser.parse_args()

    data = pd.read_csv(args.infile_path, sep="\t")
    data = data.sort_values(args.filter_column)[
        ["GeneName", args.filter_column, "logFC"]
        ]
    data = data[data[args.filter_column] <= args.filter_value][:args.number_genes]

    data.set_index("GeneName", inplace=True)

    gtf = read_gtf(args.gtf_file)
    gtf.set_index(args.gtf_column, inplace=True)

    data = data.merge(gtf, how="left", left_index=True, right_index=True)
    data.reset_index(inplace=True)
    data = data[data.columns[3:].tolist() + data.columns[:3].tolist()]
    data.end = data.start
    data.start = data.start + args.start_offset

    if args.outfile_path is not None:
        print("".join(["Writing data to: ", args.outfile_path]))
        data.to_csv(args.outfile_path, sep="\t", header=False, index=False)

if __name__ == '__main__':
    main()
