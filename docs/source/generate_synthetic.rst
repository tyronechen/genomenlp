Generate synthetic sequences for use in classification
======================================================

This explains the use of ``generate_synthetic.py``. Generates synthetic sequences given a ``fasta`` file.

Source data
-----------

Any ``fasta`` file can be used.

Results
-------

.. NOTE::

  Entry points are available if this is installed using the automated conda method. You can then use the command line argument directly, for example: ``create_dataset_bio``. If not, you will need to use the script directly, which follows the same naming pattern, for example: ``python create_dataset_bio.py``.

Running the code as below::

  python generate_synthetic.py \
    path/to/infile.fa \
    -o path/to/outfile.fa

You will obtain a ``fasta`` file with synthetic sequences generated according to your settings. By default, dinucleotide frequency is calculated **for each sequence** and used to generate a corresponding null sequence. Reverse complement is possible if needed. This can be used in two-step classification in cases where you do not have a control set.

Notes
-----

The input file can be provided in ``gzip`` format. However, output will be a plain ``text`` file as sequences are read and written line by line.

Usage
-----

::

  python generate_synthetic.py -h
  usage: generate_synthetic.py [-h] [-b BLOCK_SIZE] [-c CONTROL_DIST] [-o OUTFILE]
                               [--do_reverse_complement]
                               infile_path

  Take fasta files, generate synthetic sequences. Accepts .gz files.

  positional arguments:
    infile_path           path to fasta/gz file

  options:
    -h, --help            show this help message and exit
    -b BLOCK_SIZE, --block_size BLOCK_SIZE
                          size of block to generate synthetic sequences from as
                          negative control (DEFAULT: 2)
    -c CONTROL_DIST, --control_dist CONTROL_DIST
                          generate control distribution by [ bootstrap | frequency
                          | /path/to/file ] (DEFAULT: frequency)
    -o OUTFILE, --outfile OUTFILE
                          write synthetic sequences (DEFAULT: "out.fa")
    --do_reverse_complement
                          turn on reverse complement (DEFAULT: OFF)
