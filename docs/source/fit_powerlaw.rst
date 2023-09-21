Compare performance of different deep learning models
=====================================================

This explains the use of ``fit_powerlaw.py``. Only works on deep learning models through the ``genomicBERT`` pipeline. For more information on the method, including interpretation, please refer to the publication (`https://arxiv.org/pdf/2202.02842.pdf`_).

.. _https://arxiv.org/pdf/2202.02842.pdf: https://arxiv.org/pdf/2202.02842.pdf

Source data
-----------

Directories containing trained models from a standard ``huggingface`` or ``pytorch`` workflow can be passed in as input.

Results
-------

.. NOTE::

  Entry points are available if this is installed using the automated conda method. You can then use the command line argument directly, for example: ``create_dataset_bio``. If not, you will need to use the script directly, which follows the same naming pattern, for example: ``python create_dataset_bio.py``.

Running the code as below::

  python fit_powerlaw.py -i [ INFILE_PATH ... ] -t OUTPUT_DIR -a N

Plots will be output to the directory. A combined plot with all performance overlaid and individual performances will be generated.

Notes
-----

Interpreting the plots may not be straightforward. Please refer to the publication for more information (`https://arxiv.org/pdf/2202.02842.pdf`_).

Usage
-----

::

  python fit_powerlaw.py -h
  usage: fit_powerlaw.py [-h] [-m MODEL_PATH [MODEL_PATH ...]] [-o OUTPUT_DIR]
                         [-a ALPHA_MAX]

  Take trained model dataset and apply power law fit. Acts as a performance
  metric which is independent of data. For more information refer here:
  https://arxiv.org/pdf/2202.02842.pdf

  optional arguments:
    -h, --help            show this help message and exit
    -m MODEL_PATH [MODEL_PATH ...], --model_path MODEL_PATH [MODEL_PATH ...]
                          path to trained model directory
    -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                          path to output metrics directory (DEFAULT: same as
                          model_path)
    -a ALPHA_MAX, --alpha_max ALPHA_MAX
                          max alpha value to plot (DEFAULT: 8)

.. NOTE::

  If you are intending to download a model and the directory path matches the one on your disk, you will need to rename or remove it since it will first use local files as a priority!
