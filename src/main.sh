#!/bin/sh
# main controller script for running the pipeline
python generate_synthetic.py \
  ../data/Yeast_promoter.gz \
  -o ../data/Yeast_promoter_synthetic.fa

python tokenise.py \
  -i ../data/Yeast_promoter.gz \
  -t ../results/tmp/yeast.json

python create_dataset.py \
  ../data/Yeast_promoter.gz \
  ../data/Yeast_promoter_synthetic.fa \
  ../results/tmp/yeast.json \
  -o ../results/tmp/

python train_model.py \
  ../results/tmp/dataset.parquet \
  ../results/tmp/yeast.json \
  -o ../results/tmp/
