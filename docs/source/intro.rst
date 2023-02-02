Ziran: Genome recoding for Machine Learning Usage
=================================================

.. raw:: html

  Copyright (c) 2022 <a href="https://orcid.org/0000-0002-9207-0385">Tyrone Chen <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>, <a href="https://orcid.org/0000-0003-0181-6258">Sonika Tyagi <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a> Navya Tyagi, and Sarthak Chauhan.

Code in this repository is provided under a `MIT license`_. This documentation is provided under a `CC-BY-3.0 AU license`_.

.. _MIT license: https://opensource.org/licenses/MIT

.. _CC-BY-3.0 AU license: https://creativecommons.org/licenses/by/3.0/au/

`Visit our lab website here.`_ Contact Sonika Tyagi at `sonika.tyagi@monash.edu`_.

.. _Visit our lab website here.: https://bioinformaticslab.erc.monash.edu/

.. _sonika.tyagi@monash.edu: mailto:sonika.tyagi@monash.edu

.. NOTE::

  `The main repository is on gitlab.`_ Please submit any issues to the main gitlab repository only.

.. _The main repository is on gitlab.: https://gitlab.com/tyagilab/ziran

.. NOTE::

  No raw data is present in this repository at time of writing as manuscript(s) associated with the primary data are unpublished.

Highlights
----------

- Takes raw sequence data directly and performs classification
- Empirical tokenisation removes the need for arbitrary k-mer selection and handles out-of-vocab "words"
- Compare multiple deep learning models without the need for retraining

Install
-------

Manual install with ``conda`` or ``pip``::
  
  gensim==4.2.0
  hyperopt==0.2.7
  pandas==1.4.2
  pytorch==1.10.0
  ray==1.13.0
  scikit-learn==1.1.1
  screed==1.0.5
  seaborn==0.11.2
  sentencepiece==0.1.96
  tokenizers==0.12.1
  tqdm==4.64.0
  transformers==4.23.1
  wandb==0.13.4
  weightwatcher==0.5.9
  xgboost==1.7.1
  yellowbrick==1.3.post1


Usage
-----

1. Preprocessing
++++++++++++++++

Tokenise the biological sequence data into segments using either empirical tokenisation or conventional k-mers. Provide input data as gzipped fasta files. Empirical tokenisation is a two-step process, while in k-merisation the tokenisation and dataset creation is performed in the same operation.

Empirical tokenisation pathway::

  python tokenise_bio.py -i [INFILE_PATH ... ] -t TOKENISER_PATH
  python create_dataset_bio.py <INFILE_SEQS_1> <INFILE_SEQS_2> <TOKENISER_PATH> -c CHUNK -o OUTFILE_DIR

Conventional k-mers pathway::

  # LABEL must match INFILE_PATH! assume that one fasta file has one seq class
  python kmerise_bio.py -i [INFILE_PATH ... ] -t TOKENISER_PATH -k KMER_SIZE -l [LABEL ... ] -c CHUNK -o OUTFILE_DIR

Embedding pathway (input files here are ``csv`` created from previous step)::

  # after the empirical tokenisation pathway::
  create_embedding_bio_sp.py -i [INFILE_PATH ... ] -t TOKENISER_PATH -o OUTFILE_DIR

  # after the conventional k-mers pathway::
  create_embedding_bio_kmers.py -i [INFILE_PATH ... ] -t TOKENISER_PATH  -o OUTFILE_DIR


.. NOTE::

  The ``CHUNK`` flag can be used to partition individual sequences into smaller chunks. ``512`` is a good starting point. The ``--no_reverse_complement`` flag should be used where non-DNA sequences are used. Vocabulary size can be set with the ``--vocab_size`` flag. For generating embeddings, number of threads can be set with ``--njobs``.


2. Classification
+++++++++++++++++

Feed the data preprocessed in the previous step into the classification pipeline. Set ``freq_method`` and ``model`` combination as needed. Hyperparameter sweeping is performed by default. For non-deep learning methods, cross-validation is performed in the same operation.

Deep learning requires a ``wandb`` account set up and configured to visualise interactive plots in real time. `Please follow the instructions on wandb to configure your own account.`_

.. _Please follow the instructions on wandb to configure your own account.: https://wandb.ai/home

Frequency-based approaches::

  python freq_pipeline.py -i [INFILE_PATH ... ] --format "csv" -t TOKENISER_PATH --freq_method [ cvec | tfidf ] --model [ rf | xg ] --kfolds N --sweep_count N --metric_opt [ accuracy | f1 | precision | recall | roc_auc ] --output_dir OUTPUT_DIR

Embedding::

  python embedding_pipeline.py -i [INFILE_PATH ... ] --format "csv" -t TOKENISER_PATH --freq_method [ cvec | tfidf ] --model [ rf | xg ] --kfolds N --sweep_count N --metric_opt [ accuracy | f1 | precision | recall | roc_auc ] --output_dir OUTPUT_DIR

.. NOTE::

  ``--model_features`` can be set to reduce the number of features used. Number of threads can be set with ``--njobs``. ``--sweep_method`` can be set to change search method between ``[ grid | random ]``.

Deep learning::

  python sweep.py <TRAIN_DATA> <FORMAT> <TOKENISER_PATH> --test TEST_DATA --valid VALIDATION_DATA --hyperparameter_sweep PARAMS.JSON --entity_name WANDB_ENTITY_NAME --project_name WANDB_PROJECT_NAME --group_name WANDB_GROUP_NAME --sweep_count N --metric_opt [ eval/accuracy | eval/validation | eval/loss | eval/precision | eval/recall ] --output_dir OUTPUT_DIR

  # use the WANDB_ENTITY_NAME, WANDB_PROJECT_NAME and the best run id corresponding to the sweep
  # WANDB_GROUP_NAME should be changed to reflect the new category of runs (eg "cval")
  python cross_validate.py <TRAIN_DATA> <FORMAT> --test TEST_DATA --valid VALIDATION_DATA --entity_name WANDB_ENTITY_NAME --project_name WANDB_PROJECT_NAME --group_name WANDB_GROUP_NAME --kfolds N --config_from_run WANDB_RUN_ID --output_dir OUTPUT_DIR


.. NOTE::

  You can provide the hyperparameter search space with a ``json`` file to ``--hyperparameter_sweep``. The ``label_names`` argument here is different from previous steps and refers to the column name containing labels, not a list of class labels. Set ``--device cuda:0`` if you have ``cuda`` installed and want to use GPU.

.. raw:: html

   <details>
   <summary><a>Example hyperparameter.json file</a></summary>

.. code-block:: python

  {
    "name" : "random",
    "method" : "random",
    "metric": {
      "name": "eval/f1",
      "goal": "maximize"
    },
    "parameters" : {
      "epochs" : {
        "values" : [1, 2, 3]
      },
      "batch_size": {
          "values": [8, 16, 32, 64]
          },
      "learning_rate" :{
        "distribution": "log_uniform_values",
        "min": 0.0001,
        "max": 0.1
        },
      "weight_decay": {
          "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        }
    },
    "early_terminate": {
        "type": "hyperband",
        "s": 2,
        "eta": 3,
        "max_iter": 27
    }
  }

.. raw:: html

   </details>

Background
----------

.. `The name is a reference to a "base state"`_ which we are trying to achieve with our data representation.

.. _The name is a reference to a "base state": https://en.wikipedia.org/wiki/Ziran

*To be written*
