Get class attribution for deep learning models
==============================================

This explains the use of ``interpret.py`` for deep learning through ``genomicBERT``.

Source data
-----------

Source data is a path to a trained ``pytorch`` classifier model directory OR a ``wandb`` run.


Results
-------

.. NOTE::

  Entry points are available if this is installed using the automated conda method. You can then use the command line argument directly, for example: ``create_dataset_bio``. If not, you will need to use the script directly, which follows the same naming pattern, for example: ``python create_dataset_bio.py``.

Running the code as below:

Deep learning
+++++++++++++

Input sequences can be provided as multiple strings and/or fasta files. If a string is provided, the file name will be the first 16 characters of the string followed by a unique string. If a fasta file is provided, the file name(s) will be the fasta header. Label names must be sorted in the order of labels, eg category 1, category 2.

::

  python interpret.py <MODEL_PATH> <INPUT_SEQS ...> [TOKENISER_PATH] [OUTPUT_DIR] [LABEL_NAMES ...]


Notes
-----

More information on `transformers interpretability is available here`_.

.. _transformers interpretability is available here: https://github.com/cdpierse/transformers-interpret

Usage
-----

genomicBERT: Deep learning
++++++++++++++++++++++++++

Sequences to test for class attribution can be provided directly or as fasta files.

::

    python interpret.py -h
    usage: interpret.py [-h] [-t TOKENISER_PATH] [-o OUTPUT_DIR] [-l LABEL_NAMES [LABEL_NAMES ...]]
                        model_path input_seqs [input_seqs ...]

    Take complete classifier and calculate feature attributions.

    positional arguments:
        model_path            path to local model directory OR wandb run
        input_seqs            input sequence(s) directly and/or fasta files

    optional arguments:
        -h, --help            show this help message and exit
        -t TOKENISER_PATH, --tokeniser_path TOKENISER_PATH
                                path to tokeniser.json file to load data from
        -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                                specify path for output (DEFAULT: ./interpret_out)
        -l LABEL_NAMES [LABEL_NAMES ...], --label_names LABEL_NAMES [LABEL_NAMES ...]
                                provide label names matching order (DEFAULT: None).
