genomicBERT: Train a deep learning classifier
=============================================

This explains the use of ``train.py``. Use this if you already know what hyperparameters are needed. Otherwise use ``sweep.py``. For conventional machine learning, the sweep, train and cross validation steps are combined in one operation.

Source data
-----------

Source data is a HuggingFace ``dataset`` object as a ``csv``, ``json`` or ``parquet`` file. Specify ``--format`` accordingly.

Results
-------

.. NOTE::

  Entry points are available if this is installed using the automated conda method. You can then use the command line argument directly, for example: ``create_dataset_bio``. If not, you will need to use the script directly, which follows the same naming pattern, for example: ``python create_dataset_bio.py``.

Running the code as below::

  python train_model.py <TRAIN_DATA> <FORMAT> <TOKENISER_PATH> --test TEST_DATA --valid VALIDATION_DATA --hyperparameter_file PARAMS.JSON --entity_name WANDB_ENTITY_NAME --project_name WANDB_PROJECT_NAME --group_name WANDB_GROUP_NAME --sweep_count N --metric_opt [ eval/accuracy | eval/validation | eval/loss | eval/precision | eval/recall ] --output_dir OUTPUT_DIR --label_names labels

.. NOTE::

  Remember to provide the ``--label_names`` argument! This is ``labels`` by default (if this wasn't changed in any previous part of the pipeline).


You will obtain a ``json`` file with weights for each token. Any special tokens you add will also be present. This will be used in the next step of creating a ``HuggingFace`` compatible dataset object.

Notes
-----

The `original documentation to specify training arguments is available here`_.

.. _original documentation to specify training arguments is available here: https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/trainer#transformers.TrainingArguments

Usage
-----

The full list of arguments is truncated, and only arguments added by this package are shown. These are available on the corresponding HuggingFace ``transformers.TrainingArguments`` documentation shown above.

::

  python train.py -h

  Take HuggingFace dataset and train. Arguments match that of
  TrainingArguments, with the addition of [ train, test, valid, tokeniser_path,
  vocab_size, model, device, entity_name, project_name, group_name,
  config_from_run, metric_opt, hyperparameter_file, no_shuffle, wandb_off,
  override_output_dir ]. See: https://huggingface.co/docs/transformers/v4.19.4/
  en/main_classes/trainer#transformers.TrainingArguments

  positional arguments:
    train                 path to [ csv | csv.gz | json | parquet ] file
    format                specify input file type [ csv | json | parquet ]
    tokeniser_path        path to tokeniser.json file to load data from

  options:
    -h, --help            show this help message and exit
    --output_dir OUTPUT_DIR
                          The output directory where the model predictions and
                          checkpoints will be written. (default: None)
    --overwrite_output_dir [OVERWRITE_OUTPUT_DIR]
                          Overwrite the content of the output directory. Use
                          this to continue training if output_dir points to a
                          checkpoint directory. (default: False)
    -t TEST, --test TEST  path to [ csv | csv.gz | json | parquet ] file
                          (default: None)
    -v VALID, --valid VALID
                          path to [ csv | csv.gz | json | parquet ] file
                          (default: None)
    -m MODEL, --model MODEL
                          choose model [ distilbert | longformer ] distilbert
                          handles shorter sequences up to 512 tokens longformer
                          handles longer sequences up to 4096 tokens (DEFAULT:
                          distilbert) (default: distilbert)
    -d DEVICE, --device DEVICE
                          choose device [ cpu | cuda:0 ] (DEFAULT: detect)
                          (default: None)
    -s VOCAB_SIZE, --vocab_size VOCAB_SIZE
                          vocabulary size for model configuration (default:
                          32000)
    -f HYPERPARAMETER_FILE, --hyperparameter_file HYPERPARAMETER_FILE
                          provide torch.bin or json file of hyperparameters.
                          NOTE: if given, this overrides all
                          HfTrainingArguments! This is overridden by
                          --config_from_run! (default: )
    -e ENTITY_NAME, --entity_name ENTITY_NAME
                          provide wandb team name (if available). NOTE: has no
                          effect if wandb is disabled. (default: )
    -p PROJECT_NAME, --project_name PROJECT_NAME
                          provide wandb project name (if available). NOTE: has
                          no effect if wandb is disabled. (default: )
    -g GROUP_NAME, --group_name GROUP_NAME
                          provide wandb group name (if desired). (default:
                          train)
    -c CONFIG_FROM_RUN, --config_from_run CONFIG_FROM_RUN
                          load arguments from existing wandb run. NOTE: if
                          given, this overrides --hyperparameter_file!
                          (default: None)
    METRIC_OPT, --metric_opt METRIC_OPT
                          score to maximise [ eval/accuracy | eval/validation |
                          eval/loss | eval/precision | eval/recall ] (DEFAULT:
                          eval/f1) (default: eval/f1)
    --override_output_dir
                          override output directory (DEFAULT: OFF) (default:
                          False)
    --no_shuffle          turn off random shuffling (DEFAULT: SHUFFLE)
                          (default: True)
    --wandb_off           log training in real time online (DEFAULT: ON)
                          (default: True)

    [ADDITIONAL ARGUMENTS TRUNCATED]
