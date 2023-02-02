<!-- [![](https://flat.badgen.net/badge/DOI/10.5281%2Fzenodo.4562010/green?scale=1.5)](https://doi.org/10.5281/zenodo.4562010) -->
<!-- [![](https://flat.badgen.net/docker/size/tyronechen/multiomics/1.0.0/amd64?scale=1.5&color=black)](https://hub.docker.com/repository/docker/tyronechen/multiomics) -->
[![](https://flat.badgen.net/badge/license/MIT/cyan?scale=1.5)](https://opensource.org/licenses/MIT)
[![](https://flat.badgen.net/badge/icon/gitlab?icon=gitlab&label&color=orange&scale=1.5)](https://gitlab.com/tyagilab/seq_utils)
[![](https://flat.badgen.net/badge/icon/@tyagilab?icon=twitter&label&scale=1.5)](https://twitter.com/tyagilab)

# Ziran: Genome recoding for Machine Learning Usage

> **NOTE**: The [main repository is on gitlab](https://gitlab.com/tyagilab/ziran). Please submit any issues to the main gitlab repository only.

> **NOTE**: No raw data is present in this repository as manuscript(s) associated with the primary data are unpublished.

Copyright (c) 2022 <a href="https://orcid.org/0000-0002-9207-0385">Tyrone Chen <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>, <a href="https://orcid.org/0000-0003-0181-6258">Sonika Tyagi <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a> Navya Tyagi, and Sarthak Chauhan.

Code in this repository is provided under a [MIT license](https://opensource.org/licenses/MIT). This documentation is provided under a [CC-BY-3.0 AU license](https://creativecommons.org/licenses/by/3.0/au/).

[Visit our lab website here.](https://bioinformaticslab.erc.monash.edu/) Contact Sonika Tyagi at [sonika.tyagi@monash.edu](mailto:sonika.tyagi@monash.edu).

## Highlights

- Takes raw sequence data directly and performs classification
- Empirical tokenisation removes the need for arbitrary k-mer selection and handles out-of-vocab "words"
- Compare multiple deep learning models without the need for retraining

## Requirements

Manual install with ``conda`` or ``pip``:
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
  transformers==4.23.1
  wandb==0.13.4
  weightwatcher==0.5.9
  xgboost==1.7.1
  yellowbrick==1.3.post1
```

## Usage

Please refer to the documentation for detailed usage information.

## Acknowledgements

[This work was supported by the [MASSIVE HPC facility](www.massive.org.au) and the authors thank the HPC team at Monash eResearch Centre for their continuous personnel support. [We acknowledge and pay respects to the Elders and Traditional Owners of the land on which our 4 Australian campuses stand](https://www.monash.edu/indigenous-australians/about-us/recognising-traditional-owners).

> **NOTE**: References are listed in the introduction section.
