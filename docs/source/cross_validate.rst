Perform cross-validation
========================

This explains the use of ``cross_validate.py`` for deep learning through the ``genomicBERT`` pipeline. For conventional machine learning, the sweep, train and cross validation steps are combined in one operation.

Requirements
------------

All available via ``pip install``::

  python==3.10.4
  ray==1.13.0
  scikit-learn==1.1.1
  tokenizers==0.11.6
  tqdm==4.64.0
  transformers==4.18.0
  wandb==0.13.4
  xgboost==1.7.1
  yellowbrick==1.3.post1

Source data
-----------

Source data is a HuggingFace ``dataset`` object as a ``csv``, ``json`` or ``parquet`` file. Specify ``--format`` accordingly. ``csv`` only for non-deep learning.


Results
-------

.. NOTE::

  Entry points are available if this is installed using the automated conda method. You can then use the command line argument directly, for example: ``create_dataset_bio``. If not, you will need to use the script directly, which follows the same naming pattern, for example: ``python create_dataset_bio.py``.

Running the code as below:

Deep learning
+++++++++++++

Specify the same data, wandb project, entity and group names as used for sweeping or training. Once the best run is identified by the user, passing the run id into ``--config_from_run`` will automatically load config of the best run from ``wandb``.

::

  # use the WANDB_ENTITY_NAME, WANDB_PROJECT_NAME and the best run id corresponding to the sweep
  # WANDB_GROUP_NAME should be changed to reflect the new category of runs (eg "cval")
  python cross_validate.py <TRAIN_DATA> <FORMAT> --test TEST_DATA --valid VALIDATION_DATA --entity_name WANDB_ENTITY_NAME --project_name WANDB_PROJECT_NAME --group_name WANDB_GROUP_NAME --kfolds N --config_from_run WANDB_RUN_ID --output_dir OUTPUT_DIR

Frequency-based approaches
++++++++++++++++++++++++++

Cross-validation is carried out within the main pipeline::

  python freq_pipeline.py -i [INFILE_PATH ... ] --format "csv" -t TOKENISER_PATH --freq_method [ cvec | tfidf ] --model [ rf | xg ] --kfolds N --sweep_count N --metric_opt [ accuracy | f1 | precision | recall | roc_auc ] --output_dir OUTPUT_DIR

Embedding
+++++++++

Cross-validation is carried out within the main pipeline::

  python embedding_pipeline.py -i [INFILE_PATH ... ] --format "csv" -t TOKENISER_PATH --freq_method [ cvec | tfidf ] --model [ rf | xg ] --kfolds N --sweep_count N --metric_opt [ accuracy | f1 | precision | recall | roc_auc ] --output_dir OUTPUT_DIR

Notes
-----

The `original documentation to specify training arguments is available here`_.

.. _original documentation to specify training arguments is available here: https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/trainer#transformers.TrainingArguments

Usage
-----

Deep learning
+++++++++++++

Sweep parameters and search space should be passed in as a ``json`` file.

::

  python ../src/cross_validate.py -h
  usage: cross_validate.py [-h] [--tokeniser_path TOKENISER_PATH] [-t TEST] [-v VALID] [-m MODEL_PATH] [-d DEVICE]
                          [-s VOCAB_SIZE] [-f HYPERPARAMETER_FILE] [-l LABEL_NAMES [LABEL_NAMES ...]] [-k KFOLDS]
                          [-e ENTITY_NAME] [-g GROUP_NAME] [-p PROJECT_NAME] [-c CONFIG_FROM_RUN] [-o METRIC_OPT]
                          [--override_output_dir] [--no_shuffle] [--wandb_off]
                          train format

  Take HuggingFace dataset and perform cross validation.

  positional arguments:
    train                 path to [ csv | csv.gz | json | parquet ] file
    format                specify input file type [ csv | json | parquet ]

  optional arguments:
    -h, --help            show this help message and exit
    --tokeniser_path TOKENISER_PATH
                          path to tokeniser.json file to load data from
    -t TEST, --test TEST  path to [ csv | csv.gz | json | parquet ] file
    -v VALID, --valid VALID
                          path to [ csv | csv.gz | json | parquet ] file
    -m MODEL_PATH, --model_path MODEL_PATH
                          path to pretrained model dir. this should contain files such as [ pytorch_model.bin,
                          config.yaml, tokeniser.json, etc ]
    -d DEVICE, --device DEVICE
                          choose device [ cpu | cuda:0 ] (DEFAULT: detect)
    -s VOCAB_SIZE, --vocab_size VOCAB_SIZE
                          vocabulary size for model configuration
    -f HYPERPARAMETER_FILE, --hyperparameter_file HYPERPARAMETER_FILE
                          provide torch.bin or json file of hyperparameters. NOTE: if given, this overrides all
                          HfTrainingArguments! This is overridden by --config_from_run!
    -l LABEL_NAMES [LABEL_NAMES ...], --label_names LABEL_NAMES [LABEL_NAMES ...]
                          provide column with label names (DEFAULT: "").
    -k KFOLDS, --kfolds KFOLDS
                          run n number of kfolds (DEFAULT: 8)
    -e ENTITY_NAME, --entity_name ENTITY_NAME
                          provide wandb team name (if available).
    -g GROUP_NAME, --group_name GROUP_NAME
                          provide wandb group name (if desired).
    -p PROJECT_NAME, --project_name PROJECT_NAME
                          provide wandb project name (if available).
    -c CONFIG_FROM_RUN, --config_from_run CONFIG_FROM_RUN
                          load arguments from existing wandb run. NOTE: if given, this overrides --hyperparameter_file!
    -o METRIC_OPT, --metric_opt METRIC_OPT
                          score to maximise [ eval/accuracy | eval/validation | eval/loss | eval/precision |
                          eval/recall ] (DEFAULT: eval/f1)
    --override_output_dir
                          override output directory (DEFAULT: OFF)
    --no_shuffle          turn off random shuffling (DEFAULT: SHUFFLE)
    --wandb_off           run hyperparameter tuning using the wandb api and log training in real time online (DEFAULT:
                          ON)