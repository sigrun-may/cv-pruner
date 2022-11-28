# Copyright (c) 2022 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from typing import List

import optuna
from optuna.pruners import BasePruner


class MultiPrunerDelegate(BasePruner):
    def __init__(self, pruner_list: List[BasePruner], prune_eager: bool = True):
        """Initializer.

        Args:
            pruner_list: List of pruners to delegate to.
            prune_eager: if ``True`` then ``prune`` will return as soon as one pruner returns ``True``.
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
