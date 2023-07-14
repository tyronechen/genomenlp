Create a dataset object from sequences (NLP version)
====================================================

This explains the use of ``create_dataset_nlp.py``. We generate a ``HuggingFace`` dataset object given a ``csv`` file containing sequences which can be of multiple categories, a pretrained ``tokeniser`` from ``tokeniser.py``, and column names corresponding to the ``title``, ``labels`` and ``sequence`` of each entry in the corpus.

Source data
-----------

Any ``csv`` file can be used, and can hold more than one category of samples. Sample input data files will be available in ``data/``. Tokeniser can be generated with ``tokenise.py``.

Results
-------

.. NOTE::

  Entry points are available if this is installed using the automated conda method. You can then use the command line argument directly, for example: ``create_dataset_bio``. If not, you will need to use the script directly, which follows the same naming pattern, for example: ``python create_dataset_bio.py``.

Running the code as below::

  python create_dataset.py \
    /path/to/input.csv \
    /path/to/tokeniser.json \
    title \
    labels \
    content \
    -o /path/to/results/

``HuggingFace`` dataset files will be written to disk. This can be loaded directly into a "conventional" deep learning pipeline.

Notes
-----

It is possible to split the dataset into chunks of n-length. This is useful when the length of individual sequences become too large to fit in memory. A sequence length of 256-512 units can effectively fit on most modern GPUs. Sequence chunks are treated as independent samples of the same class and no merging of weights is performed in this implementation. Note that ``create_dataset_bio.py`` and ``create_dataset_nlp.py`` workflows are structured differently to account for the differences in conventional biological vs human language corpora, but the processes are conceptually identical.

More information on the HuggingFace 🤗 ``Dataset`` object `is available online`_.

.. _is available online: https://huggingface.co/docs/datasets/package_reference/main_classes

Usage
-----

::

  python create_dataset_nlp.py -h
  usage: create_dataset_nlp.py [-h] [-d CONTROL_DIST] [-o OUTFILE_DIR]
                               [-s SPECIAL_TOKENS [SPECIAL_TOKENS ...]] [-c CHUNK]
                               [--split_train SPLIT_TRAIN] [--split_test SPLIT_TEST]
                               [--split_val SPLIT_VAL] [--no_shuffle]
                               infile_path tokeniser_path title labels content

  Take control and test csv files, tokeniser and convert to HuggingFace🤗 dataset object. csv files can be
  .gz.

  positional arguments:
    infile_path           path to csv/gz file
    tokeniser_path        load tokeniser file
    title                 name of the column in the csv file which contains a unique identifier
    labels                name of the column in the csv file which contains labels
    content               name of the column in the csv file which contains content

  optional arguments:
    -h, --help            show this help message and exit
    -d CONTROL_DIST, --control_dist CONTROL_DIST
                          supply category 2
    -o OUTFILE_DIR, --outfile_dir OUTFILE_DIR
                          write 🤗 dataset to directory as [ csv | json | parquet | dir/ ] (DEFAULT:
                          "hf_out/")
    -s SPECIAL_TOKENS [SPECIAL_TOKENS ...], --special_tokens SPECIAL_TOKENS [SPECIAL_TOKENS ...]
                          assign special tokens, eg space and pad tokens (DEFAULT: ["<s>", "</s>",
                          "<unk>", "<pad>", "<mask>"])
    -c CHUNK, --chunk CHUNK
                          split seqs into n-length blocks (DEFAULT: None)
    --split_train SPLIT_TRAIN
                          proportion of training data (DEFAULT: 0.90)
    --split_test SPLIT_TEST
                          proportion of testing data (DEFAULT: 0.05)
    --split_val SPLIT_VAL
                          proportion of validation data (DEFAULT: 0.05)
    --no_shuffle          turn off shuffle for data split (DEFAULT: ON)
