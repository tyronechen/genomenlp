# README

Tokenising the entire human genome `GRCh38.p14` requires exponentially increasing memory resources.

As a workaround, the genome was split into multiple contig/strings for tokenisation, before being recombined at the end.

- Each genomic sequence was split into contig/string lengths of 2^9
- Each subfile contained 2^16 contig/strings
- The final subsequence and subfile may be smaller than the others as all contigs are preserved. Given the vast quantity of contigs available, the impact of this is expected to be insignificant

The tokenisation process has several properties which require a multi-stage recombination process:

- During tokenisation, a specific vocabulary size was specified
- At the same time, weights were assigned per unique token

Therefore, to reconstitute the tokeniser:

1. All tokens were combined across all subfiles
2. Unique tokens were preserved as-is
3. Non-unique token weights were pooled via mean

Multiple combinations of vocabulary size were applied.
Numeric quantities in file names reflect the associated vocabulary size in the original *subfile tokenisation*
Therefore, the actual number of tokens in the final file do not correspond to these quantities.

Vocab size:

```
Subfile | Primary 
--------|--------
800     | 3829
1600    | 8473
3200    | 18657
4800    | 30839
6400    | 45403
9600    | 80263
12800   | 120853
32000   | 463800
320000  | 2154913
```

Tokenisers in the form of `json` files for a range of vocabulary sizes are provided for convenience.
Users may select an appropriate vocabulary size resolution for their use case, or customise the size to their own settings.
Slotting the tokeniser into the `genomenlp` pipeline to replace a standard tokeniser file is possible as-is.
However, note that the same tokeniser should be used throughout the pipeline.
For transparency, token weight distributions (unitless) and corresponding visualisations are included for inspection.
For simplicity, we suggest attempting smaller tokenisers such as `9600` or `12800` as a starting point.

# Sample code for reproducibility

```
# obtain GRCh38.p14.zip from NCBI

# split genomic sequence into contigs and subfiles
python parse_grch38p14.py

# perform tokenisation per subfile
vocab="800
1600
3200
4800
6400
9600
12800
32000
320000"
for i in out_segments/*gz; do
    for j in ${vocab}; do
        tokenise_bio -i $1 -t $1_$2.json -v $2;
    done
done

# reconstitute master token file
vocabs="800
1600
3200
4800
6400
9600
12800
32000
320000"
merges="inner
outer"

for vocab in ${vocabs}; do
  for merge in ${merges}; do
    compare_empirical_tokens *_${vocab}.json \
      -t pooled_mean_${merge}.${vocab}.json \
      -w pooled_mean_${merge}_${vocab} \
      -m ${merge} \
      -p mean \
      -o pooled_mean_${merge}_${vocab}.pdf
  done
done
```
