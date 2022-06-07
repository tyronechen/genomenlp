#!/usr/bin/python
# take top N differentially expressed genes and find coordinates for bedtools
import argparse
from gtfparse import read_gtf
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description='Offset start, end coordinates of an input bed-like file.'
    )
    parser.add_argument('infile_path', type=str, help='path to DEG table')
    parser.add_argument('-c', '--filter_column', type=int, default=4,
                        help='column to apply filter value (DEFAULT: 4)')
    parser.add_argument('-f', '--filter_value', type=int, default=1000,
                        help='filter out anything below this (DEFAULT: 1000)')
    parser.add_argument('-o', '--outfile_path', type=str, default=None,
                        help='write bedfile to this path (DEFAULT: None)')
    parser.add_argument('-s', '--start_offset', type=int, default=-1000,
                        help='extract region N bases upstream (DEFAULT: -1000)')

    args = parser.parse_args()

    data = pd.read_csv(args.infile_path, sep="\t", header=None)
    col = int(args.filter_column)
    data[["end"]] = data[[col]] - args.start_offset
    data = data[data["end"] > args.filter_value]
    data[3] = data[3].str.replace("forward", "+")
    data[3] = data[3].str.replace("reverse", "-")
    data = data[[0, 4, "end", 1, 8, 3]]
    data.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand"]
    print(data)
    if args.outfile_path is not None:
        print("".join(["Writing data to: ", args.outfile_path]))
        data.to_csv(args.outfile_path, sep="\t", header=False, index=False)

if __name__ == '__main__':
    main()
