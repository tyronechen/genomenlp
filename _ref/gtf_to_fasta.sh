#!/bin/bash
# run gtf -> bed -> fasta pipeline
which gtf2bed

# genome fasta file
genome=$1
# annotation gtf file
i=$2
# length upstream of TSS
len=$3
cpu=1

# unzip
echo ${genome}
gzip -dv ${genome}

# get rid of anything that goes off the end of the chromosome
echo ${i} ${i}.${len}.bed.tmp
gtf2bed < ${i} | \
  awk -v len=${len} -F'\t' '{if($2<=len) next; if($2) print}' | \
  awk -v len=${len} -F'\t' 'BEGIN{OFS="\t"} { $3=$2; $2=$2-len ; print }' > \
  ${i}.${len}.bed.tmp

# offset -1000 of TSS to mine for information-rich regions
# echo bedops --range -${len}:-${len} --everything ${i}.${len}.bed.tmp ${i}.${len}.bed
# bedops --range -${len}:-${len} --everything ${i}.${len}.bed.tmp > \
#   ${i}.${len}.bed

# it is possible that this may overlap into genic regions but informative
echo bedops --element-of 1 ${i}.${len}.bed.tmp ${i}.${len}.bed ${i}.${len}.bed.overlap
bedops --element-of 1 ${i}.${len}.bed.tmp ${i}.${len}.bed > \
  ${i}.${len}.bed.overlap
wc -l ${i} ${i}.${len}.bed ${i}.${len}.bed.overlap

# extract the seqs we need, reverse complementing reverse strands
echo bedtools getfasta -s -bed ${i}.${len}.bed.tmp -fi ${genome/.gz} -fo ${i}.${len}.fasta
bedtools getfasta -s -bed ${i}.${len}.bed.tmp -fi ${genome/.gz} -fo ${i}.${len}.fasta
rm ${i}.${len}.bed.tmp

# unzip
echo ${genome/.gz}
gzip -v ${genome/.gz}

python convert_input.py ${i}.${len}.fasta \
  -t ${cpu} \
  -o ${i}.${len}.bed \
  -s 100000 \
  -i

gzip ${i}.${len}.fasta
