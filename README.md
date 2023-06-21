<!-- [![](https://flat.badgen.net/badge/DOI/10.5281%2Fzenodo.4562010/green?scale=1.5)](https://doi.org/10.5281/zenodo.4562010) -->
<!-- [![](https://flat.badgen.net/docker/size/tyronechen/multiomics/1.0.0/amd64?scale=1.5&color=black)](https://hub.docker.com/repository/docker/tyronechen/multiomics) -->
[![](https://readthedocs.org/projects/genomenlp/badge/?version=latest&style=for-the-badge)](https://genomenlp.readthedocs.io/en/latest/)
[![](https://flat.badgen.net/badge/license/MIT/cyan?scale=1.5)](https://opensource.org/licenses/MIT)
[![](https://flat.badgen.net/badge/icon/gitlab?icon=gitlab&label&color=orange&scale=1.5)](https://gitlab.com/tyagilab/genomenlp)
[![](https://flat.badgen.net/badge/icon/github?icon=github&label&color=black&scale=1.5)](https://github.com/tyronechen/genomenlp)
[![](https://flat.badgen.net/badge/icon/@tyagilab?icon=twitter&label&scale=1.5)](https://twitter.com/tyagilab)
[![Anaconda-Server Badge](https://anaconda.org/tyronechen/genomenlp/badges/version.svg)](https://anaconda.org/tyronechen/genomenlp)
[![Anaconda-Server Badge](https://anaconda.org/tyronechen/genomenlp/badges/platforms.svg)](https://anaconda.org/tyronechen/genomenlp)
[![Anaconda-Server Badge](https://anaconda.org/tyronechen/genomenlp/badges/downloads.svg)](https://anaconda.org/tyronechen/genomenlp)

# genomeNLP: Genome recoding for Machine Learning Usage incorporating genomicBERT

> **NOTE**: The [main repository is on github](https://github.com/tyronechen/genomenlp) but is also mirrored on [gitlab](https://gitlab.com/tyagilab/genomenlp). Please submit any issues to the main github repository only.

Copyright (c) 2022 <a href="https://orcid.org/0000-0002-9207-0385">Tyrone Chen <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>, <a href="https://orcid.org/0000-0002-8797-3168">Navya Tyagi <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>, Sarthak Chauhan, <a href="https://orcid.org/0000-0002-2296-2126">Anton Y. Peleg <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>, and <a href="https://orcid.org/0000-0003-0181-6258">Sonika Tyagi <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>.

Code in this repository is provided under a [MIT license](https://opensource.org/licenses/MIT). This documentation is provided under a [CC-BY-3.0 AU license](https://creativecommons.org/licenses/by/3.0/au/).

[Visit our lab website here.](https://bioinformaticslab.erc.monash.edu/) Contact Sonika Tyagi at [sonika.tyagi@monash.edu](mailto:sonika.tyagi@monash.edu).

## Highlights

- We provide a comprehensive classification of genomic data tokenisation and representation approaches for ML applications along with their pros and cons.
- Using our ``genomicBERT`` deep learning pipeline, we infer k-mers directly from the data and handle out-of-vocabulary words. At the same time, we achieve a significantly reduced vocabulary size compared to the conventional k-mer approach reducing the computational complexity drastically.
- Our method is agnostic to species or biomolecule type as it is data-driven.
- We enable comparison of trained model performance without requiring original input data, metadata or hyperparameter settings.
- We present the first publicly available, high-level toolkit that infers the grammar of genomic data directly through artificial neural networks.
- Preprocessing, hyperparameter sweeps, cross validations, metrics and interactive visualisations are automated but can be adjusted by the user as needed.

![graphical abstract describing the repository](fig/graphical_abstract.png)

## Cite us with:

*Will be provided on publication*

## Install

### Conda (automated)

This is the recommended install method as it automatically handles dependencies. Note that this has only been tested on a linux operating system.

First try this:

```
conda install -c tyronechen genomenlp
```

If there are any errors with the previous step (especially if you are on a cluster with GPU access), try this first and then repeat the previous step:

```
conda install -c anaconda cudatoolkit
```

If neither works, please submit an issue with the full stack trace and any supporting information.

### Conda (manual)

Clone the git repository. This will also allow you to manually run the python scripts.

Then manually install the following dependencies with ``conda`` or ``pip``:

```
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
```

You should then be able to run the scripts manually from ``src/genomenlp``. As with the automated step, ``cudatoolkit`` may be required.

## Usage

Please refer to the documentation for detailed usage information of the package and the ``genomicBERT`` pipeline.

[![](https://readthedocs.org/projects/genomenlp/badge/?version=latest&style=for-the-badge)](https://genomenlp.readthedocs.io/en/latest/)

## Acknowledgements

TC was supported by an Australian Government Research Training Program (RTP) Scholarship and Monash Faculty of Science Dean’s Postgraduate Research Scholarship. ST acknowledges support from Early Mid-Career Fellowship by Australian Academy of Science and Australian Women Research Success Grant at Monash University. AP and ST acnowledge MRFF funding for the SuperbugAI flagship. This work was supported by the [MASSIVE HPC facility](www.massive.org.au) and the authors thank the Monash Bioinformatics Platform as well as the HPC team at Monash eResearch Centre for their continuous personnel support. We thank Yashpal Ramakrishnaiah for helpful suggestions on package management, code architecture and documentation hosting. We thank Jane Hawkey for advice on recovering deprecated bacterial protein identifier mappings in NCBI. We thank Andrew Perry and Richard Lupat for helping resolve an issue with the python package building process. Biorender was used to create many figures in this publication. [We acknowledge and pay respects to the Elders and Traditional Owners of the land on which our 4 Australian campuses stand](https://www.monash.edu/indigenous-australians/about-us/recognising-traditional-owners).

<!-- > **NOTE**: References are listed in the introduction section. -->
