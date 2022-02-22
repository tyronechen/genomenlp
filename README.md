<!-- [![](https://flat.badgen.net/badge/DOI/10.5281%2Fzenodo.4562010/green?scale=1.5)](https://doi.org/10.5281/zenodo.4562010) -->
<!-- [![](https://flat.badgen.net/docker/size/tyronechen/multiomics/1.0.0/amd64?scale=1.5&color=black)](https://hub.docker.com/repository/docker/tyronechen/multiomics) -->
[![](https://flat.badgen.net/badge/license/MIT/cyan?scale=1.5)](https://opensource.org/licenses/MIT)
[![](https://flat.badgen.net/badge/icon/gitlab?icon=gitlab&label&color=orange&scale=1.5)](https://gitlab.com/tyagilab/seq_utils)
[![](https://flat.badgen.net/badge/icon/@tyagilab?icon=twitter&label&scale=1.5)](https://twitter.com/tyagilab)

# Genome recoding for Machine Learning Usage

> **NOTE**: The [main repository is on gitlab](https://gitlab.com/tyagilab/seq_utils). It is [also mirrored on github](https://github.com/tyronechen/seq_utils) but please submit any issues to the main gitlab repository only.

> **NOTE**: No raw data is present in this repository as manuscript(s) associated with the primary data are unpublished.

Copyright (c) 2022 <a href="https://orcid.org/0000-0002-9207-0385">Tyrone Chen <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>, <a href="https://orcid.org/0000-0003-0181-6258">Sonika Tyagi <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>

Code in this repository is provided under a [MIT license](https://opensource.org/licenses/MIT). This documentation is provided under a [CC-BY-3.0 AU license](https://creativecommons.org/licenses/by/3.0/au/).

[Visit our lab website here.](https://bioinformaticslab.erc.monash.edu/) Contact Sonika Tyagi at [sonika.tyagi@monash.edu](mailto:sonika.tyagi@monash.edu).

## Background

## Aim

Compile a python package for recoding genomic sequences into a format (1 hot encoding, k-mers, … ) that can be easily passed into deep learning frameworks. Ideally something with minimal dependencies that can just be imported and used alongside popular machine learning frameworks (pytorch).

Generic usage (this is just an example only, not restricted to this specific workflow):

```
import RecodeSeqs
seqs = RecodeSeqs(a_BioPython_seq_object)
seqs.recode_onehot(padding=TRUE)
seqs.recode_kmers(length=500)  # break seqs into n-length substrings
                               # you can think of this as tokenising
seqs.recode_something…         # other encodings as we think of them
```

## Problem

There are a lot of packages which encode image and natural language data into `python` objects for input into popular deep learning frameworks such as `pytorch` and `tensorflow`. There are no known packages which do the same thing for biological sequence data.

For example, genomic data are commonly represented as `bam`, `bed`, `fasta`, `fastq`, `sam`, `gtf`, `gff` files, or other genomic representations. In some cases, the raw sequence may not even be inherently present and require further processing (eg in the case of `bed` files, it records coordinates which need to be matched to a reference file or database to extract the actual data) or if we want to encode fasta sequences into images or text. (Descriptions of these file formats are publicly available.)

Sample metadata is also inconsistent even within file formats of the same version. For example, in `fasta/q` files, naming conventions are inconsistent. In `sam/bam/gff/gtf` files, arbitrary fields can be added. For this part, it is impossible to account for all possible cases, so these should be specified in a constant format as minimum input requirement. Alternatively, metadata can be ignored and a specific metadata format can be created by us to restrict input.

## Planning

- Recode on the fly (probably start with this) or generate a binary object for easier loading next time (can worry about this later)? Eg hdf5 or TFRecord file
- …

## Relevant literature

- Evaluation of Convolutionary Neural Networks Modeling of DNA Sequences using Ordinal versus one-hot Encoding Method
- …

## Relevant software

- `Biopython` (read and parse fastq files, possible to read line by line instead of loading whole file into memory, very important considering fasta/q file sizes!)
- Existing code for 1-hot encoding for use with CNN (currently loops through fastq files and feeds them to `Pytorch`, if you go back to earlier commits you can see the `Tensorflow` version)
- ksahlin/strobemers: A repository for generating strobemers and evalaution
- https://en.wikipedia.org/wiki/N-gram
- https://github.com/huggingface/transformers/tree/v4.10.2
- http://interpret.ml/
- https://erroranalysis.ai/
- https://2-bitbio.com/2018/06/one-hot-encode-dna-sequence-using.html
- https://www.jwilber.me/permutationtest/
- ...

## Notes

Different architectures have different input types. `Tensorflow` uses `TFRecord` objects on disk, need specify own encoder and decoder. `Pytorch` more flexible, probably easier to use, recommend starting here. Not sure about `megvii`.
A lot of ChIP-Seq and ATAC-Seq protocols use genome tiling where they break the genome into lots of contigs/substrings of N length. May be a good place to start finding working code and fork from there.


## Acknowledgements

[This work was supported by the [MASSIVE HPC facility](www.massive.org.au) and the authors thank the HPC team at Monash eResearch Centre for their continuous personnel support. [We acknowledge and pay respects to the Elders and Traditional Owners of the land on which our 4 Australian campuses stand](https://www.monash.edu/indigenous-australians/about-us/recognising-traditional-owners).

> **NOTE**: References are listed in the introduction section.
