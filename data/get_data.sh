#!/bin/sh
echo "This downloads the source data of the latest version of regulonDB only!"
echo "For more information please visit: https://regulondb.ccg.unam.mx"

# note that the file format may change!
wget --no-check-certificate \
  'https://regulondb.ccg.unam.mx/menu/download/datasets/files/PromoterSet.txt'

# convert to fasta file using regulondb identifiers as headers
grep -v '#' PromoterSet.txt | cut -f1,6 | sed "s|^|>|" | tr '\t' '\n' > \
  PromoterSet.fa
