# CV-Pruner

[![MIT License](https://img.shields.io/github/license/sigrun-may/cv-pruner)](https://github.com/sigrun-may/cv-pruner/blob/main/LICENSE)
[![pytest](https://github.com/sigrun-may/cv-pruner/actions/workflows/pytest.yml/badge.svg)](https://github.com/sigrun-may/cv-pruner/actions/workflows/pytest.yml)
[![Static Code Checks](https://github.com/sigrun-may/cv-pruner/actions/workflows/static_checks.yml/badge.svg)](https://github.com/sigrun-may/cv-pruner/actions/workflows/static_checks.yml)
[![GitHub issues](https://img.shields.io/github/issues-raw/sigrun-may/cv-pruner)](https://github.com/sigrun-may/cv-pruner/issues)

Nested cross-validation is necessary to avoid biased model performance in embedded feature selection in high-dimensional data with tiny sample sizes. Standard pruning algorithms to accelerate hyperparameter optimization must prune late or risk aborting computations of promising hyperparameter sets due to high variance in the performance evaluation metric. The cv-pruner allows combining a comparison-based pruning strategy with two additional pruning strategies based on domain or prior knowledge. One of them prunes semantically meaningless trials. The other is a threshold-based pruning strategy that extrapolates the performance evaluation metric. The combination of pruning strategies can lead to a massive speedup in computation. 

## Installation

CV-Pruner is available at [the Python Package Index (PyPI)](https://pypi.org/project/cv-pruner/).
It can be installed with pip:

```bash
$ pip install cv-pruner
```

## Licensing

Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)<br/>
Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften

Licensed under the **MIT License** (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License by reviewing the file
[LICENSE](https://github.com/sigrun-may/cv-pruner/blob/main/LICENSE) in the repository.
