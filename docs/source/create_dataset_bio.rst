Create a dataset object from sequences
======================================

This explains the use of ``create_dataset_bio.py``. We generate a ``HuggingFace`` dataset object given a ``fasta`` file containing sequences, a ``fasta`` file containing control sequences, and a pretrained ``tokeniser`` from ``tokeniser.py``. The dataset can then enter the ``genomicBERT`` pipeline.

Source data
-----------

Any ``fasta`` file can be used, with each ``fasta`` file representing a sequence collection of one category. Sample input data files will be available in ``data/``. If needed, control data can be generated with ``generate_synthetic.py``. Tokeniser can be generated with ``tokenise.py``.

Results
-------

.. NOTE::

  Entry points are available if this is installed using the automated conda method. You can then use the command line argument directly, for example: ``create_dataset_bio``. If not, you will need to use the script directly, which follows the same naming pattern, for example: ``python create_dataset_bio.py``.

Running the code as below::

  python create_dataset_bio.py <INFILE_SEQS_1> <INFILE_SEQS_2> <TOKENISER_PATH> -c CHUNK -o OUTFILE_DIR

``HuggingFace``-like dataset files will be written to disk. This can be loaded directly into a "conventional" deep learning pipeline.

Notes
-----

It is possible to split the dataset into chunks of n-length. This is useful when the length of individual sequences become too large to fit in memory. A sequence length of 256-512 units can effectively fit on most modern GPUs. Sequence chunks are treated as independent samples of the same class and no merging of weights is performed in this implementation. Note that ``create_dataset_bio.py`` and ``create_dataset_nlp.py`` workflows are structured differently to account for the differences in conventional biological vs human language corpora, but the processes are conceptually identical.

More information on the HuggingFace 🤗 ``Dataset`` object `is available online`_.

.. _is available online: https://huggingface.co/docs/datasets/package_reference/main_classes

Usage
-----

::

  python create_dataset_bio.py -h
  usage: create_dataset_bio.py [-h] [-o OUTFILE_DIR] [-s SPECIAL_TOKENS [SPECIAL_TOKENS ...]] [-c CHUNK]
                               [--split_train SPLIT_TRAIN] [--split_test SPLIT_TEST]
                               [--split_val SPLIT_VAL] [--no_reverse_complement] [--no_shuffle]
                               infile_path control_dist tokeniser_path

  Take control and test fasta files, tokeniser and convert to HuggingFace🤗 dataset object. Fasta files
  can be .gz. Sequences are reverse complemented by default.

  positional arguments:
    infile_path           path to fasta/gz file
    control_dist          supply control seqs
    tokeniser_path        load tokeniser file

  optional arguments:
    -h, --help            show this help message and exit
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
    --no_reverse_complement
                          turn off reverse complement (DEFAULT: ON)
    --no_shuffle          turn off shuffle for data split (DEFAULT: ON)
