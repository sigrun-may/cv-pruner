# Copyright (c) 2022 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import datetime
from typing import List, Optional, Union

import numpy as np
import optuna
from optuna.pruners import BasePruner


class MultiPrunerDelegate(BasePruner):
    """Bundle and delegate to multiple Optuna pruners."""

    def __init__(self, pruner_list: List[BasePruner], prune_eager: bool = True):
        """Initializer.

        Args:
            pruner_list: List of pruners to delegate to.
            prune_eager: If ``True`` then ``prune`` will return as soon as one pruner returns ``True``.
                If ``False`` then we iterate all pruners and then return ``True`` if
                at least one of them returned ``True``.
        """
        self.pruner_list = pruner_list
        self.prune_eager = prune_eager

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        pruner_results = []
        for pruner in self.pruner_list:
            pruner_result = pruner.prune(study, trial)
            if self.prune_eager and pruner_result:
                return True
            else:
                pruner_results.append(pruner_result)
        return any(pruner_results)


class NoFeatureSelectedPruner(BasePruner):
    def __init__(self):
        self.feature_values = None

    def communicate_feature_values(self, feature_values: Union[np.ndarray, List[float]]):
        self.feature_values = feature_values

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        if self.feature_values is None:
            raise RuntimeError("'feature_values' is not set! Did you forget to call 'communicate_feature_values'?")

        if isinstance(self.feature_values, list):
            self.feature_values = np.array(self.feature_values)

        return np.sum(self.feature_values != 0) == 0


class BenchmarkPruneFunctionWrapper(BasePruner):
    def __init__(self, pruner: BasePruner, pruner_name: Optional[str] = None):
        self.pruner = pruner
        self.prune_reported = False
        if pruner_name is None:
            pruner_name = self.pruner.__class__.__name__
        self.pruner_name = pruner_name

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        prune_result = self.pruner.prune(study, trial)
        if prune_result and not self.prune_reported:
            pruning_timestamp = datetime.datetime.now()  # TODO: check if this is right!
            intermediate_values = trial.intermediate_values.values()
            step = len(intermediate_values)
            trial.set_user_attr(f"{self.pruner_name}_pruned_at", {"step": step, "timestamp": pruning_timestamp})
            self.prune_reported = True

        return prune_result
