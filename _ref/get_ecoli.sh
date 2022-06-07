#!/bin/bash
# download ecoli data
wget 'http://regulondb.ccg.unam.mx/menu/download/datasets/files/E_coli_K12_MG1655_U00096.3.txt'
wget 'http://regulondb.ccg.unam.mx/menu/download/datasets/files/PromoterSet.txt'
wget 'http://regulondb.ccg.unam.mx/menu/download/datasets/files/BindingSiteSet.txt'

grep -v '#' PromoterSet.txt > PromoterSet.tmp
sed -e "s|^|E_coli_K12_MG1655_U00096.3       |" PromoterSet.tmp > PromoterSet.bed
python offset_bed.py PromoterSet.bed -c 4 -o "Ecoli_K12_upstream_promoters.bed"
bedtools getfasta -fi E_coli_K12_MG1655_U00096.3.txt -bed Ecoli_K12_upstream_promoters.bed -fo Ecoli_K12_upstream_promoters.fasta
