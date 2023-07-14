Create embeddings from a tokenised dataset
==========================================

This explains the use of ``create_embedding_bio_sp.py`` and ``create_embedding_bio_kmers.py``. Only use this if you plan to use embeddings directly.

Source data
-----------

Use ``csv`` files created from either ``create_dataset_bio.py`` or ``kmerise_bio.py``.

Results
-------

.. NOTE::

  Entry points are available if this is installed using the automated conda method. You can then use the command line argument directly, for example: ``create_dataset_bio``. If not, you will need to use the script directly, which follows the same naming pattern, for example: ``python create_dataset_bio.py``.

Empirical tokenisation
++++++++++++++++++++++

::

  create_embedding_bio_sp.py -i [INFILE_PATH ... ] -t TOKENISER_PATH -o OUTFILE_DIR

Conventional k-mers
+++++++++++++++++++

::

  create_embedding_bio_kmers.py -i [INFILE_PATH ... ] -t TOKENISER_PATH  -o OUTFILE_DIR

The resulting output will be used in ``embedding_pipeline.py``.

Notes
-----

Embeddings are generated for each individual token. For example::

  # original seq of category X
  AAAAACCCCCTTTTTGGGGG

  # split into tokens using desired method
  [AAAAA]
  [CCCCC]
  ...

  # each token gets projected onto an embedding
  [0.1 0.2 0.3 ...]
  [0.3 0.4 0.5 ...]
  ...

Usage
-----

Empirical tokenisation
++++++++++++++++++++++

::

  python create_embedding_bio_sp.py -h
  usage: create_embedding_bio_sp.py [-h] [-i INFILE_PATH [INFILE_PATH ...]]
                                    [-o OUTPUT_DIR] [-c COLUMN_NAMES]
                                    [-l LABELS] [-x COLUMN_NAME] [-m MODEL]
                                    [-t TOKENISER_PATH]
                                    [-s SPECIAL_TOKENS [SPECIAL_TOKENS ...]]
                                    [-n NJOBS] [--w2v_min_count W2V_MIN_COUNT]
                                    [--w2v_sg W2V_SG]
                                    [--w2v_vector_size W2V_VECTOR_SIZE]
                                    [--w2v_window W2V_WINDOW]
                                    [--no_reverse_complement]
                                    [--sample_seq SAMPLE_SEQ]

  Take fasta files, tokeniser and generate embedding. Fasta files can be .gz.
  Sequences are reverse complemented by default.

  options:
    -h, --help            show this help message and exit
    -i INFILE_PATH [INFILE_PATH ...], --infile_path INFILE_PATH [INFILE_PATH ...]
                          path to fasta/gz file
    -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                          write embeddings to disk (DEFAULT: "embed/")
    -c COLUMN_NAMES, --column_names COLUMN_NAMES
                          column name for sp tokenised data (DEFAULT:
                          input_str)
    -l LABELS, --labels LABELS
                          column name for data labels (DEFAULT: labels)
    -x COLUMN_NAME, --column_name COLUMN_NAME
                          column name for extracting embeddings (DEFAULT:
                          input_str)
    -m MODEL, --model MODEL
                          load existing model (DEFAULT: None)
    -t TOKENISER_PATH, --tokeniser_path TOKENISER_PATH
                          load tokenised data
    -s SPECIAL_TOKENS [SPECIAL_TOKENS ...], --special_tokens SPECIAL_TOKENS [SPECIAL_TOKENS ...]
                          assign special tokens, eg space and pad tokens
                          (DEFAULT: ["<s>", "</s>", "<unk>", "<pad>",
                          "<mask>"])
    -n NJOBS, --njobs NJOBS
                          set number of threads to use
    --w2v_min_count W2V_MIN_COUNT
                          set minimum count for w2v (DEFAULT: 1)
    --w2v_sg W2V_SG       0 for bag-of-words, 1 for skip-gram (DEFAULT: 1)
    --w2v_vector_size W2V_VECTOR_SIZE
                          set w2v matrix dimensions (DEFAULT: 100)
    --w2v_window W2V_WINDOW
                          set context window size for w2v (DEFAULT: -/+10)
    --no_reverse_complement
                          turn off reverse complement (DEFAULT: ON)
    --sample_seq SAMPLE_SEQ
                          project sample sequence on embedding (DEFAULT: None)

Conventional k-mers
+++++++++++++++++++

::

  python create_embedding_bio_kmers.py -h
  usage: create_embedding_bio_kmers.py [-h] [-i INFILE_PATH [INFILE_PATH ...]]
                                       [-o OUTPUT_DIR] [-m MODEL] [-k KSIZE]
                                       [-w SLIDE] [-c CHUNK] [-n NJOBS]
                                       [-s SAMPLE_SEQ] [-v VOCAB_SIZE]
                                       [--w2v_min_count W2V_MIN_COUNT]
                                       [--w2v_sg W2V_SG]
                                       [--w2v_vector_size W2V_VECTOR_SIZE]
                                       [--w2v_window W2V_WINDOW]
                                       [--no_reverse_complement]

  Take tokenised data, parameters and generate embedding. Note that this takes
  output of kmerise_bio.py, and NOT raw fasta files.

  options:
    -h, --help            show this help message and exit
    -i INFILE_PATH [INFILE_PATH ...], --infile_path INFILE_PATH [INFILE_PATH ...]
                          path to input tokenised data file
    -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                          write embeddings to disk (DEFAULT: "embed/")
    -m MODEL, --model MODEL
                          load existing model (DEFAULT: None)
    -k KSIZE, --ksize KSIZE
                          set size of k-mers
    -w SLIDE, --slide SLIDE
                          set length of sliding window on k-mers (min 1)
    -c CHUNK, --chunk CHUNK
                          split seqs into n-length blocks (DEFAULT: None)
    -n NJOBS, --njobs NJOBS
                          set number of threads to use
    -s SAMPLE_SEQ, --sample_seq SAMPLE_SEQ
                          set sample sequence to test model (DEFAULT: None)
    -v VOCAB_SIZE, --vocab_size VOCAB_SIZE
                          vocabulary size for model config (DEFAULT: all)
    --w2v_min_count W2V_MIN_COUNT
                          set minimum count for w2v (DEFAULT: 1)
    --w2v_sg W2V_SG       0 for bag-of-words, 1 for skip-gram (DEFAULT: 1)
    --w2v_vector_size W2V_VECTOR_SIZE
                          set w2v matrix dimensions (DEFAULT: 100)
    --w2v_window W2V_WINDOW
                          set context window size for w2v (DEFAULT: -/+10)
    --no_reverse_complement
                          turn off reverse complement (DEFAULT: ON)
