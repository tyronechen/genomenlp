#!/bin/bash
# download and preprocess DNA data for promoter and non-promoter sequences
# original github: https://github.com/khanhlee/bert-promoter/
wget 'https://raw.githubusercontent.com/khanhlee/bert-promoter/main/data/non_promoter.fasta'
wget 'https://raw.githubusercontent.com/khanhlee/bert-promoter/main/data/promoter.fasta'
gzip *fasta
