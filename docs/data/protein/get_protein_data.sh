#!/bin/bash
# download and preprocess protein data for RNA and DNA binding proteins
# original article: https://doi.org/10.1016/j.jmb.2020.09.008
wget 'http://bliulab.net/iDRBP_MMC/static/dataset/training_dataset.txt'
wget 'http://bliulab.net/iDRBP_MMC/static/dataset/test_dataset_TEST474.txt'
wget 'http://bliulab.net/iDRBP_MMC/static/dataset/test_dataset_PDB255.txt'

csplit --digits=2 --quiet --prefix=outfile training_dataset.txt "/------------------------------------------------------------/+1" "{*}"
sed '$d' outfile02 | sed '$d' > train_dna_binding.fa
sed '$d' outfile04 | sed '$d' > train_rna_binding.fa
rm outfile0*

csplit --digits=2 --quiet --prefix=outfile test_dataset_TEST474.txt "/------------------------------------------------------------/+1" "{*}"
sed '$d' outfile02 | sed '$d' > test_TEST474_dna_binding.fa
sed '$d' outfile04 | sed '$d' > test_TEST474_rna_binding.fa
rm outfile0*

csplit --digits=2 --quiet --prefix=outfile test_dataset_PDB255.txt "/------------------------------------------------------------/+1" "{*}"
sed '$d' outfile02 | sed '$d' > test_PDB255_dna_binding.fa
sed '$d' outfile04 | sed '$d' > test_PDB255_rna_binding.fa
rm outfile0*

# we combine the full dataset and later repartition it with the pipeline
cat train_dna_binding.fa test_TEST474_dna_binding.fa test_PDB255_dna_binding.fa > dna_binding.fa
cat train_rna_binding.fa test_TEST474_rna_binding.fa test_PDB255_rna_binding.fa > rna_binding.fa

gzip *fa

