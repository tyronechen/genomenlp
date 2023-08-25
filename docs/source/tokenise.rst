Create a token set from sequences
=================================

This explains the use of ``kmerise_bio.py`` and ``tokenise_bio.py``. In ``tokenise_bio.py`` we empirically derive tokens from biological sequence data which can be used in downstream applications such as ``genomicBERT``.

Source data
-----------

Any ``fasta`` file can be used (nucleic acid or protein).

Results
-------

.. NOTE::

  Entry points are available if this is installed using the automated conda method. You can then use the command line argument directly, for example: ``create_dataset_bio``. If not, you will need to use the script directly, which follows the same naming pattern, for example: ``python create_dataset_bio.py``.

Running the code as below:

Empirical tokenisation
++++++++++++++++++++++

::

  python tokenise_bio.py -i [ INFILE_PATH ... ] -t TOKENISER_PATH

You will obtain a ``json`` file with weights for each token. Any special tokens you add will also be present. This will be used in the next step of creating a ``HuggingFace`` compatible dataset object.

Conventional k-mers
+++++++++++++++++++

::

  python kmerise_bio.py -i [INFILE_PATH ... ] -t TOKENISER_PATH -k KMER_SIZE -l [LABEL ... ] -c CHUNK -o OUTFILE_DIR

For k-mers, ``HuggingFace``-like dataset files will be written to disk in the same operation. This can be loaded directly into a "conventional" deep learning pipeline.

However, the file is not split into partitions. You can use it directly if you already have other partitions corresponding to training, testing and validation data. If not, you will need to create a dataset in the next stage, using the ``tokeniser.json`` file generated in this step. 

Notes
-----

Please refer to `HuggingFace tokenisers`_ for more detailed information:

.. _HuggingFace tokenisers: https://github.com/huggingface/tokenizers

Usage
-----

Empirical tokenisation
++++++++++++++++++++++

For empirical tokenisation, the next step is to run ``create_dataset_bio.py``. Reverse complementing ``Y/R`` is supported.

::

  python tokenise.py -h
  usage: tokenise.py [-h] [-i INFILE_PATHS [INFILE_PATHS ...]] [-t TOKENISER_PATH]
                     [-s SPECIAL_TOKENS [SPECIAL_TOKENS ...]] [-e EXAMPLE_SEQ]

  Take gzip fasta file(s), run empirical tokenisation and export json.

  options:
    -h, --help            show this help message and exit
    -i INFILE_PATHS [INFILE_PATHS ...], --infile_paths INFILE_PATHS [INFILE_PATHS ...]
                          path to files with biological seqs split by line
    -t TOKENISER_PATH, --tokeniser_path TOKENISER_PATH
                          path to tokeniser.json file to save or load data
    -v VOCAB_SIZE, --vocab_size VOCAB_SIZE
                          select vocabulary size (DEFAULT: 32000)
    -b BREAK_SIZE, --break_size BREAK_SIZE
                          split long reads, keep all by default (DEFAULT: None)
    -c CASE, --case CASE  change case, retain original by default (DEFAULT: None)
    -s SPECIAL_TOKENS [SPECIAL_TOKENS ...], --special_tokens SPECIAL_TOKENS [SPECIAL_TOKENS ...]
                          assign special tokens, eg space and pad tokens 
                          (DEFAULT: ["<s>", "</s>", "<unk>", "<pad>", "<mask>"])
    -e EXAMPLE_SEQ, --example_seq EXAMPLE_SEQ
                          show token to seq map for a sequence (DEFAULT: None)

Conventional k-mers
+++++++++++++++++++

Note that this step also generates a dataset object in the same operation. Reverse complementing ``Y/R`` is supported.

Here we take a list of infile paths, and a list of matching labels. Eg ``--infile_path file1.fasta file2.fasta``, then ``--label 0 1``.

::

  python ../src/kmerise_bio.py -h
  usage: kmerise_bio.py [-h] [-i INFILE_PATH [INFILE_PATH ...]]
                        [-o OUTFILE_PATH] [-c CHUNK] [-m MAPPINGS]
                        [-t TOKENISER_PATH] [-k KMER_SIZE]
                        [-l LABEL [LABEL ...]] [--no_reverse_complement]

  Take gzip fasta file(s), kmerise reads and export csv.

  options:
    -h, --help            show this help message and exit
    -i INFILE_PATH [INFILE_PATH ...], --infile_path INFILE_PATH [INFILE_PATH ...]
                          path to files with biological seqs split by line
    -o OUTFILE_PATH, --outfile_path OUTFILE_PATH
                          path to output huggingface-like dataset.csv file
    -c CHUNK, --chunk CHUNK
                          split seqs into n-length blocks (DEFAULT: None)
    -m MAPPINGS, --mappings MAPPINGS
                          path to output mappings file
    -t TOKENISER_PATH, --tokeniser_path TOKENISER_PATH
                          path to tokeniser.json file to save data
    -k KMER_SIZE, --kmer_size KMER_SIZE
                          split seqs into n-length blocks (DEFAULT: None)
    -l LABEL [LABEL ...], --label LABEL [LABEL ...]
                          provide integer label for seqs (order must match
                          infile_path!)
    --no_reverse_complement
                          turn off reverse complement (DEFAULT: ON)
