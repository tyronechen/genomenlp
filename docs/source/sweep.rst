Perform a hyperparameter sweep
==============================

This explains the use of ``sweep.py`` for machine and deep learning through ``genomicBERT``. If you already know what hyperparameters are needed, you can use ``train_model.py``. For conventional machine learning, the sweep, train and cross validation steps are combined in one operation.

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

::

  python sweep.py <TRAIN_DATA> <FORMAT> <TOKENISER_PATH> --test TEST_DATA --valid VALIDATION_DATA --hyperparameter_sweep PARAMS.JSON --entity_name WANDB_ENTITY_NAME --project_name WANDB_PROJECT_NAME --group_name WANDB_GROUP_NAME --sweep_count N --metric_opt [ eval/accuracy | eval/validation | eval/loss | eval/precision | eval/recall ] --output_dir OUTPUT_DIR

Frequency-based approaches
++++++++++++++++++++++++++

::

  python freq_pipeline.py -i [INFILE_PATH ... ] --format "csv" -t TOKENISER_PATH --freq_method [ cvec | tfidf ] --model [ rf | xg ] --kfolds N --sweep_count N --metric_opt [ accuracy | f1 | precision | recall | roc_auc ] --output_dir OUTPUT_DIR

Embedding
+++++++++

::

  python embedding_pipeline.py -i [INFILE_PATH ... ] --format "csv" -t TOKENISER_PATH --freq_method [ cvec | tfidf ] --model [ rf | xg ] --kfolds N --sweep_count N --metric_opt [ accuracy | f1 | precision | recall | roc_auc ] --output_dir OUTPUT_DIR

Notes
-----

The `original documentation to specify training arguments is available here`_.

.. _original documentation to specify training arguments is available here: https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/trainer#transformers.TrainingArguments

Usage
-----

genomicBERT: Deep learning
++++++++++++++++++++++++++

Sweep parameters and search space should be passed in as a ``json`` file.

.. raw:: html

   <details>
   <summary><a>Example hyperparameter.json file</a></summary>

.. code-block:: json

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

::

  usage: sweep.py [-h] [-t TEST] [-v VALID] [-m MODEL]
                  [--model_features MODEL_FEATURES] [-o OUTPUT_DIR] [-d DEVICE]
                  [-s VOCAB_SIZE] [-w HYPERPARAMETER_SWEEP]
                  [-l LABEL_NAMES [LABEL_NAMES ...]] [-n SWEEP_COUNT]
                  [-e ENTITY_NAME] [-p PROJECT_NAME] [-g GROUP_NAME]
                  [-c METRIC_OPT] [-r RESUME_SWEEP] [--fp16_off] [--wandb_off]
                  train format tokeniser_path

  Take HuggingFace dataset and perform parameter sweeping.

  positional arguments:
    train                 path to [ csv | csv.gz | json | parquet ] file
    format                specify input file type [ csv | json | parquet ]
    tokeniser_path        path to tokeniser.json file to load data from

  options:
    -h, --help            show this help message and exit
    -t TEST, --test TEST  path to [ csv | csv.gz | json | parquet ] file
    -v VALID, --valid VALID
                          path to [ csv | csv.gz | json | parquet ] file
    -m MODEL, --model MODEL
                          choose model [ distilbert | longformer ] distilbert
                          handles shorter sequences up to 512 tokens longformer
                          handles longer sequences up to 4096 tokens (DEFAULT:
                          distilbert)
    --model_features MODEL_FEATURES
                          number of features in data to use (DEFAULT: ALL)
                          NOTE: this is separate from the vocab_size argument.
                          under normal circumstances (eg a tokeniser generated
                          by tokenise_bio), setting this is not necessary
    -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                          specify path for output (DEFAULT: ./sweep_out)
    -d DEVICE, --device DEVICE
                          choose device [ cpu | cuda:0 ] (DEFAULT: detect)
    -s VOCAB_SIZE, --vocab_size VOCAB_SIZE
                          vocabulary size for model configuration
    -w HYPERPARAMETER_SWEEP, --hyperparameter_sweep HYPERPARAMETER_SWEEP
                          run a hyperparameter sweep with config from file
    -l LABEL_NAMES [LABEL_NAMES ...], --label_names LABEL_NAMES [LABEL_NAMES ...]
                          provide column with label names (DEFAULT: "").
    -n SWEEP_COUNT, --sweep_count SWEEP_COUNT
                          run n hyperparameter sweeps (DEFAULT: 64)
    -e ENTITY_NAME, --entity_name ENTITY_NAME
                          provide wandb team name (if available).
    -p PROJECT_NAME, --project_name PROJECT_NAME
                          provide wandb project name (if available).
    -g GROUP_NAME, --group_name GROUP_NAME
                          provide wandb group name (if desired).
    METRIC_OPT, --metric_opt METRIC_OPT
                          score to maximise [ eval/accuracy | eval/validation |
                          eval/loss | eval/precision | eval/recall ] (DEFAULT:
                          eval/f1)
    -r RESUME_SWEEP, --resume_sweep RESUME_SWEEP
                          provide sweep id to resume sweep.
    --fp16_off            turn fp16 off for precision / cpu (DEFAULT: ON)
    --wandb_off           run hyperparameter tuning using the wandb api and log
                          training in real time online (DEFAULT: ON)

Frequency based approach
++++++++++++++++++++++++

::

  python freq_pipeline.py -h
  usage: freq_pipeline.py [-h] [--infile_path INFILE_PATH [INFILE_PATH ...]]
                          [--format FORMAT] [--embeddings EMBEDDINGS]
                          [--chunk_size CHUNK_SIZE] [-t TOKENISER_PATH]
                          [-f FREQ_METHOD] [--column_names COLUMN_NAMES]
                          [--column_name COLUMN_NAME] [-m MODEL]
                          [-e MODEL_FEATURES] [-k KFOLDS]
                          [--ngram_from NGRAM_FROM] [--ngram_to NGRAM_TO]
                          [--split_train SPLIT_TRAIN] [--split_test SPLIT_TEST]
                          [--split_val SPLIT_VAL] [-o OUTPUT_DIR]
                          [-s VOCAB_SIZE]
                          [--special_tokens SPECIAL_TOKENS [SPECIAL_TOKENS ...]]
                          [-w HYPERPARAMETER_SWEEP]
                          [--sweep_method SWEEP_METHOD] [-n SWEEP_COUNT]
                          [-c METRIC_OPT] [-j NJOBS] [-d PRE_DISPATCH]

  Take HuggingFace dataset and perform parameter sweeping.

  options:
    -h, --help            show this help message and exit
    --infile_path INFILE_PATH [INFILE_PATH ...]
                          path to [ csv | csv.gz | json | parquet ] file
    --format FORMAT       specify input file type [ csv | json | parquet ]
    --embeddings EMBEDDINGS
                          path to embeddings model file
    --chunk_size CHUNK_SIZE
                          iterate over input file for these many rows
    -t TOKENISER_PATH, --tokeniser_path TOKENISER_PATH
                          path to tokeniser.json file to load data from
    -f FREQ_METHOD, --freq_method FREQ_METHOD
                          choose dist [ cvec | tfidf ] (DEFAULT: tfidf)
    --column_names COLUMN_NAMES
                          column name for sp tokenised data (DEFAULT:
                          input_str)
    --column_name COLUMN_NAME
                          column name for extracting embeddings (DEFAULT:
                          input_str)
    -m MODEL, --model MODEL
                          choose model [ rf | xg ] (DEFAULT: rf)
    -e MODEL_FEATURES, --model_features MODEL_FEATURES
                          number of features in data to use (DEFAULT: ALL)
    -k KFOLDS, --kfolds KFOLDS
                          number of cross validation folds (DEFAULT: 8)
    --ngram_from NGRAM_FROM
                          ngram slice starting index (DEFAULT: 1)
    --ngram_to NGRAM_TO   ngram slice ending index (DEFAULT: 1)
    --split_train SPLIT_TRAIN
                          proportion of training data (DEFAULT: 0.90)
    --split_test SPLIT_TEST
                          proportion of testing data (DEFAULT: 0.05)
    --split_val SPLIT_VAL
                          proportion of validation data (DEFAULT: 0.05)
    -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                          specify path for output (DEFAULT: ./results_out)
    -s VOCAB_SIZE, --vocab_size VOCAB_SIZE
                          vocabulary size for model configuration
    --special_tokens SPECIAL_TOKENS [SPECIAL_TOKENS ...]
                          assign special tokens, eg space and pad tokens
                          (DEFAULT: ["<s>", "</s>", "<unk>", "<pad>",
                          "<mask>"])
    -w HYPERPARAMETER_SWEEP, --hyperparameter_sweep HYPERPARAMETER_SWEEP
                          run a hyperparameter sweep with config from file
    --sweep_method SWEEP_METHOD
                          specify sweep search strategy [ bayes | grid | random
                          ] (DEFAULT: random)
    -n SWEEP_COUNT, --sweep_count SWEEP_COUNT
                          run n hyperparameter sweeps (DEFAULT: 8)
    -c METRIC_OPT, --metric_opt METRIC_OPT
                          score to maximise [ accuracy | f1 | precision |
                          recall ] (DEFAULT: f1)
    -j NJOBS, --njobs NJOBS
                          run on n threads (DEFAULT: -1)
    -d PRE_DISPATCH, --pre_dispatch PRE_DISPATCH
                          specify dispatched jobs (DEFAULT: 0.5*n_jobs)

Embedding based approach
++++++++++++++++++++++++

::

  python embedding_pipeline.py -h
  usage: embedding_pipeline.py [-h]
                               [--infile_path INFILE_PATH [INFILE_PATH ...]]
                               [--format FORMAT] [--embeddings EMBEDDINGS]
                               [--chunk_size CHUNK_SIZE] [-t TOKENISER_PATH]
                               [-f FREQ_METHOD] [--column_names COLUMN_NAMES]
                               [--column_name COLUMN_NAME] [-m MODEL]
                               [-e MODEL_FEATURES] [-k KFOLDS]
                               [--ngram_from NGRAM_FROM] [--ngram_to NGRAM_TO]
                               [--split_train SPLIT_TRAIN]
                               [--split_test SPLIT_TEST]
                               [--split_val SPLIT_VAL] [-o OUTPUT_DIR]
                               [-s VOCAB_SIZE]
                               [--special_tokens SPECIAL_TOKENS [SPECIAL_TOKENS ...]]
                               [-w HYPERPARAMETER_SWEEP]
                               [--sweep_method SWEEP_METHOD] [-n SWEEP_COUNT]
                               [-c METRIC_OPT] [-j NJOBS] [-d PRE_DISPATCH]

  Take HuggingFace dataset and perform parameter sweeping.

  options:
    -h, --help            show this help message and exit
    --infile_path INFILE_PATH [INFILE_PATH ...]
                          path to [ csv | csv.gz | json | parquet ] file
    --format FORMAT       specify input file type [ csv | json | parquet ]
    --embeddings EMBEDDINGS
                          path to embeddings model file
    --chunk_size CHUNK_SIZE
                          iterate over input file for these many rows
    -t TOKENISER_PATH, --tokeniser_path TOKENISER_PATH
                          path to tokeniser.json file to load data from
    -f FREQ_METHOD, --freq_method FREQ_METHOD
                          choose dist [ embed ] (DEFAULT: embed)
    --column_names COLUMN_NAMES
                          column name for sp tokenised data (DEFAULT:
                          input_str)
    --column_name COLUMN_NAME
                          column name for extracting embeddings (DEFAULT:
                          input_str)
    -m MODEL, --model MODEL
                          choose model [ rf | xg ] (DEFAULT: rf)
    -e MODEL_FEATURES, --model_features MODEL_FEATURES
                          number of features in data to use (DEFAULT: ALL)
    -k KFOLDS, --kfolds KFOLDS
                          number of cross validation folds (DEFAULT: 8)
    --ngram_from NGRAM_FROM
                          ngram slice starting index (DEFAULT: 1)
    --ngram_to NGRAM_TO   ngram slice ending index (DEFAULT: 1)
    --split_train SPLIT_TRAIN
                          proportion of training data (DEFAULT: 0.90)
    --split_test SPLIT_TEST
                          proportion of testing data (DEFAULT: 0.05)
    --split_val SPLIT_VAL
                          proportion of validation data (DEFAULT: 0.05)
    -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                          specify path for output (DEFAULT: ./results_out)
    -s VOCAB_SIZE, --vocab_size VOCAB_SIZE
                          vocabulary size for model configuration
    --special_tokens SPECIAL_TOKENS [SPECIAL_TOKENS ...]
                          assign special tokens, eg space and pad tokens
                          (DEFAULT: ["<s>", "</s>", "<unk>", "<pad>",
                          "<mask>"])
    -w HYPERPARAMETER_SWEEP, --hyperparameter_sweep HYPERPARAMETER_SWEEP
                          run a hyperparameter sweep with config from file
    --sweep_method SWEEP_METHOD
                          specify sweep search strategy [ bayes | grid | random
                          ] (DEFAULT: random)
    -n SWEEP_COUNT, --sweep_count SWEEP_COUNT
                          run n hyperparameter sweeps (DEFAULT: 8)
    -c METRIC_OPT, --metric_opt METRIC_OPT
                          score to maximise [ accuracy | f1 | precision |
                          recall ] (DEFAULT: f1)
    -j NJOBS, --njobs NJOBS
                          run on n threads (DEFAULT: -1)
    -d PRE_DISPATCH, --pre_dispatch PRE_DISPATCH
                          specify dispatched jobs (DEFAULT: 0.5*n_jobs)
