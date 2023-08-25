genomeNLP: Genome recoding for Machine Learning Usage incorporating genomicBERT
===============================================================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8135590.svg
  :target: https://doi.org/10.5281/zenodo.8135590

.. image:: https://anaconda.org/tyronechen/genomenlp/badges/license.svg   
  :target: https://anaconda.org/tyronechen/genomenlp

.. image:: https://anaconda.org/tyronechen/genomenlp/badges/version.svg   
  :target: https://anaconda.org/tyronechen/genomenlp

.. image:: https://anaconda.org/tyronechen/genomenlp/badges/downloads.svg   
  :target: https://anaconda.org/tyronechen/genomenlp  


.. raw:: html


  Copyright (c) 2022 <a href="https://orcid.org/0000-0002-9207-0385">Tyrone Chen <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>, <a href="https://orcid.org/0000-0002-8797-3168">Navya Tyagi <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>, Sarthak Chauhan, <a href="https://orcid.org/0000-0002-2296-2126">Anton Y. Peleg <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>, and <a href="https://orcid.org/0000-0003-0181-6258">Sonika Tyagi <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>.

Code in this repository is provided under a `MIT license`_. This documentation is provided under a `CC-BY-3.0 AU license`_.

.. _MIT license: https://opensource.org/licenses/MIT

.. _CC-BY-3.0 AU license: https://creativecommons.org/licenses/by/3.0/au/

`Visit our lab website here.`_ Contact Sonika Tyagi at `sonika.tyagi@monash.edu`_.

.. _Visit our lab website here.: https://bioinformaticslab.erc.monash.edu/

.. _sonika.tyagi@monash.edu: mailto:sonika.tyagi@monash.edu

.. NOTE::

  `The main repository is on github`_ but also mirrored on gitlab. Please submit any issues to the main github repository only.

.. _The main repository is on github: https://github.com/tyronechen/genomenlp


Highlights
----------

- We provide a comprehensive classification of genomic data tokenisation and representation approaches for ML applications along with their pros and cons.
- Using our ``genomicBERT`` deep learning pipeline, we infer k-mers directly from the data and handle out-of-vocabulary words. At the same time, we achieve a significantly reduced vocabulary size compared to the conventional k-mer approach reducing the computational complexity drastically.
- Our method is agnostic to species or biomolecule type as it is data-driven.
- We enable comparison of trained model performance without requiring original input data, metadata or hyperparameter settings.
- We present the first publicly available, high-level toolkit that infers the grammar of genomic data directly through artificial neural networks.
- Preprocessing, hyperparameter sweeps, cross validations, metrics and interactive visualisations are automated but can be adjusted by the user as needed.

.. image:: ../../fig/graphical_abstract.png
   :alt: graphical abstract describing the repository

Cite us with:
-------------

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8135590.svg
   :target: https://doi.org/10.5281/zenodo.8135590

Manuscript::

  @article{chen2023genomicbert,
    title={genomicBERT and data-free deep-learning model evaluation},
    author={Chen, Tyrone and Tyagi, Navya and Chauhan, Sarthak and Peleg, Anton Y and Tyagi, Sonika},
    journal={bioRxiv},
    month={jun},
    pages={2023--05},
    year={2023},
    publisher={Cold Spring Harbor Laboratory},
    doi={10.1101/2023.05.31.542682},
    url={https://doi.org/10.1101/2023.05.31.542682}
}

Software::

  @software{chen_tyrone_2023_8143218,
  author       = {Chen, Tyrone and
                  Tyagi, Navya and
                  Chauhan, Sarthak and
                  Peleg, Anton Y. and
                  Tyagi, Sonika},
  title        = {{genomicBERT and data-free deep-learning model 
                   evaluation}},
  month        = jul,
  year         = 2023,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {latest},
  doi          = {10.5281/zenodo.8135590},
  url          = {https://doi.org/10.5281/zenodo.8135590}
  }

Install
-------

Mamba (automated)
+++++++++++++++++

This is the recommended install method as it automatically handles dependencies. Note that this has only been tested on a linux operating system.

.. NOTE::

  Installing with ``mamba`` is highly recommended. Installing with ``pip`` will not work. Installing with ``conda`` will be slow. `You can find instructions for setting up mamba here`_. Please submit any issues to the main github repository only.

.. _You can find instructions for setting up mamba here: https://mamba.readthedocs.io/en/latest/installation.html


First try this::

  mamba install -c conda-forge -c tyronechen genomenlp

If there are any errors with the previous step (especially if you are on a cluster with GPU access), try this first and then repeat the previous step::

  mamba install -c anaconda cudatoolkit

If neither works, please submit an issue with the full stack trace and any supporting information.


Mamba (manual)
++++++++++++++

Clone the git repository. This will also allow you to manually run the python scripts.

Then manually install the following dependencies with ``mamba``. Installing with ``pip`` will not work as some distributions are not available on ``pip``.::

  gensim==4.2.0
  hyperopt==0.2.7
  matplotlib==3.5.2
  pandas==1.4.2
  pytorch==1.10.0
  ray==1.13.0
  scikit-learn==1.1.1
  screed==1.0.5
  seaborn==0.11.2
  sentencepiece==0.1.96
  tokenizers==0.12.1
  tqdm==4.64.0
  transformers==4.30.0
  wandb==0.13.4
  weightwatcher==0.5.9
  xgboost==1.7.1
  yellowbrick==1.3.post1

You should then be able to run the scripts manually from ``src/genomenlp``. As with the automated step, ``cudatoolkit`` may be required.


Usage
-----

0. Available commands
+++++++++++++++++++++

If installed correctly using the automated ``mamba`` method, each of these commands will be available directly on the command line::

  create_dataset_bio
  create_dataset_nlp
  create_embedding_bio_sp
  create_embedding_bio_kmers
  cross_validate
  embedding_pipeline
  fit_powerlaw
  freq_pipeline
  generate_synthetic
  interpret
  kmerise_bio
  parse_sp_tokens
  summarise_metrics
  sweep
  tokenise_bio
  train

If installed correctly using the manual ``mamba`` method, each of the above commands can be called from their corresponding python script, for example::

  python create_dataset_bio.py


1. Preprocessing
++++++++++++++++

Tokenise the biological sequence data into segments using either empirical tokenisation or conventional k-mers. Provide input data as gzipped fasta files. Empirical tokenisation is a two-step process, while in k-merisation the tokenisation and dataset creation is performed in the same operation. Both methods generate data compatible with the ``genomicBERT`` pipeline.

Empirical tokenisation pathway::

  tokenise_bio -i [INFILE_PATH ... ] -t TOKENISER_PATH
  create_dataset_bio <INFILE_SEQS_1> <INFILE_SEQS_2> <TOKENISER_PATH> -c CHUNK -o OUTFILE_DIR

Conventional k-mers pathway::

  # LABEL must match INFILE_PATH! assume that one fasta file has one seq class
  kmerise_bio -i [INFILE_PATH ... ] -t TOKENISER_PATH -k KMER_SIZE -l [LABEL ... ] -c CHUNK -o OUTFILE_DIR
  create_dataset_bio <INFILE_SEQS_1> <INFILE_SEQS_2> <TOKENISER_PATH> -c CHUNK -o OUTFILE_DIR

Embedding pathway (input files here are ``csv`` created from previous step)::

  # after the empirical tokenisation pathway::
  create_embedding_bio_sp -i [INFILE_PATH ... ] -t TOKENISER_PATH -o OUTFILE_DIR

  # after the conventional k-mers pathway::
  create_embedding_bio_kmers -i [INFILE_PATH ... ] -t TOKENISER_PATH  -o OUTFILE_DIR


.. NOTE::

  The ``CHUNK`` flag can be used to partition individual sequences into smaller chunks. ``512`` is a good starting point. The ``--no_reverse_complement`` flag should be used where non-DNA sequences are used. Vocabulary size can be set with the ``--vocab_size`` flag. For generating embeddings, number of threads can be set with ``--njobs``.


2. Classification
+++++++++++++++++

Feed the data preprocessed in the previous step into the classification pipeline. Set ``freq_method`` and ``model`` combination as needed. Hyperparameter sweeping is performed by default. For non-deep learning methods, cross-validation is performed in the same operation.

Deep learning with the ``genomicBERT`` pipeline requires a ``wandb`` account set up and configured to visualise interactive plots in real time. `Please follow the instructions on wandb to configure your own account.`_

.. _Please follow the instructions on wandb to configure your own account.: https://wandb.ai/home

Frequency-based approaches::

  freq_pipeline -i [INFILE_PATH ... ] --format "csv" -t TOKENISER_PATH --freq_method [ cvec | tfidf ] --model [ rf | xg ] --kfolds N --sweep_count N --metric_opt [ accuracy | f1 | precision | recall | roc_auc ] --output_dir OUTPUT_DIR

Embedding::

  embedding_pipeline -i [INFILE_PATH ... ] --format "csv" -t TOKENISER_PATH --freq_method [ cvec | tfidf ] --model [ rf | xg ] --kfolds N --sweep_count N --metric_opt [ accuracy | f1 | precision | recall | roc_auc ] --output_dir OUTPUT_DIR

.. NOTE::

  ``--model_features`` can be set to reduce the number of features used. Number of threads can be set with ``--njobs``. ``--sweep_method`` can be set to change search method between ``[ grid | random ]``.

``genomicBERT`` deep learning pipeline::

  sweep <TRAIN_DATA> <FORMAT> <TOKENISER_PATH> --test TEST_DATA --valid VALIDATION_DATA --hyperparameter_sweep PARAMS.JSON --entity_name WANDB_ENTITY_NAME --project_name WANDB_PROJECT_NAME --group_name WANDB_GROUP_NAME --sweep_count N --metric_opt [ eval/accuracy | eval/validation | eval/loss | eval/precision | eval/recall ] --output_dir OUTPUT_DIR

  # use the WANDB_ENTITY_NAME, WANDB_PROJECT_NAME and the best run id corresponding to the sweep
  # WANDB_GROUP_NAME should be changed to reflect the new category of runs (eg "cval")
  cross_validate <TRAIN_DATA> <FORMAT> --test TEST_DATA --valid VALIDATION_DATA --entity_name WANDB_ENTITY_NAME --project_name WANDB_PROJECT_NAME --group_name WANDB_GROUP_NAME --kfolds N --config_from_run WANDB_RUN_ID --output_dir OUTPUT_DIR


.. NOTE::

  You can provide the hyperparameter search space with a ``json`` file to ``--hyperparameter_sweep``. The ``label_names`` argument here is different from previous steps and refers to the column name containing labels, not a list of class labels. Set ``--device cuda:0`` if you have ``cuda`` installed and want to use GPU.

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

3. Comparing deep learning models trained by genomicBERT
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The included method only works on deep learning models, including those trained through the ``genomicBERT`` pipeline. For more information on the method, including interpretation, please refer to the publication (`https://arxiv.org/pdf/2202.02842.pdf`_).

.. _https://arxiv.org/pdf/2202.02842.pdf: https://arxiv.org/pdf/2202.02842.pdf

::

  fit_powerlaw -i [ INFILE_PATH ... ] -t OUTPUT_DIR -a N

4. Case study
+++++++++++++

A detailed case study is available for reference.


Background
----------

*To be written*


Acknowledgements
----------------

TC was supported by an Australian Government Research Training Program (RTP) Scholarship and Monash Faculty of Science Dean’s Postgraduate Research Scholarship. ST acknowledges support from Early Mid-Career Fellowship by Australian Academy of Science and Australian Women Research Success Grant at Monash University. AP and ST acnowledge MRFF funding for the SuperbugAI flagship. This work was supported by the MASSIVE HPC facility (https://www.massive.org.au) and the authors thank the Monash Bioinformatics Platform as well as the HPC team at Monash eResearch Centre for their continuous personnel support. We thank Yashpal Ramakrishnaiah for helpful suggestions on package management, code architecture and documentation hosting. We thank Jane Hawkey for advice on recovering deprecated bacterial protein identifier mappings in NCBI. We thank Andrew Perry and Richard Lupat for helping resolve an issue with the python package building process. Biorender was used to create many figures in this publication. We acknowledge and pay respects to the Elders and Traditional Owners of the land on which our 4 Australian campuses stand (https://www.monash.edu/indigenous-australians/about-us/recognising-traditional-owners).
