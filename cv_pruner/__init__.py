# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""CV-Pruner main package."""

from cv_pruner.cv_pruner import Method, should_prune_against_threshold, no_features_selected


__version__ = "0.0.1rc2"

__all__ = [
    "should_prune_against_threshold",
    "no_features_selected",
    "Method",
    "__version__",
]
