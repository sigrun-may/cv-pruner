# Copyright (c) 2022 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import datetime
import sys
from typing import Any, List, Optional, Union

import numpy as np
import optuna
from optuna.pruners import BasePruner
from optuna.study import StudyDirection

from cv_pruner import Method, should_prune_against_threshold


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
            pruning_timestamp = datetime.datetime.now().isoformat(timespec="microseconds")  # like Optuna
            intermediate_values = trial.intermediate_values.values()
            step = len(intermediate_values)
            trial.set_user_attr(f"{self.pruner_name}_pruned_at", {"step": step, "timestamp": pruning_timestamp})
            self.prune_reported = True

        return prune_result


# TODO: should optimal_metric_value be part of the initializer?
class RepeatedTrainingThresholdPruner(BasePruner):
    """Pruner to detect trials with insufficient performance.

    Prune if an extrapolated or cumulated metric exceeds upper threshold,
    falls behind lower threshold or reaches ``nan``.

    Args:
        threshold:
            A minimum value which determines whether pruner prunes or not.
            If an extrapolated value is smaller than lower, it prunes.
            Or a maximum value which determines whether pruner prunes or not.
            If an extrapolated value is larger than upper, it prunes.
        n_warmup_steps:
            Pruning is disabled if the number of reported steps is less than the given number of warmup steps.
        active_until_step:
            Pruning will be disabled if the number of reported steps is greater than the
            given number of ``active_until_step``.
        extrapolation_interval:
            Cross-validation folds or folds of inner cross-validation in case of nested cross-validation.
            If no value has been reported at the time of a pruning check, that particular check
            will be postponed until a value is reported. Value must be at least 1.
    """

    def __init__(
            self,
            threshold: float,
            n_warmup_steps: int = 3,
            active_until_step: int = sys.maxsize,
            extrapolation_interval: int = 1,
    ) -> None:

        threshold = _check_value(threshold)

        if n_warmup_steps < 0:
            raise ValueError("Number of warmup steps cannot be negative but got {}.".format(n_warmup_steps))
        if extrapolation_interval < 1:
            raise ValueError("Cross-validation folds must be at least 1 but got {}.".format(extrapolation_interval))

        if not n_warmup_steps > 2:
            raise ValueError("n_warmup_steps must be greater than 2!")
        if not active_until_step > n_warmup_steps:
            raise ValueError("active_until_step must be greater than n_warmup_steps!")

        self._threshold = threshold
        self._n_warmup_steps = n_warmup_steps
        self._active_until_step = active_until_step
        self._cross_validation_folds = extrapolation_interval

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:

        # no values have been reported
        step = trial.last_step
        if step is None:
            return False

        intermediate_values = trial.intermediate_values.values()
        reported_step_count = len(intermediate_values)

        # do not prune if reported_step_count is outside n_warmup_steps and active_until_step
        if reported_step_count < self._n_warmup_steps or reported_step_count > self._active_until_step:
            return False

        return should_prune_against_threshold(
            folds_inner_cv=self._cross_validation_folds,
            validation_metric_history=list(intermediate_values),
            threshold_for_pruning=self._threshold,
            start_step=3,  # set min value here to let the pruner decide before
            stop_step=sys.maxsize,  # set max value here to let the pruner decide before
            direction_to_optimize_is_minimize=study.direction == StudyDirection.MINIMIZE,
            method=Method.OPTIMAL_METRIC,
            optimal_metric_value=0,
        )


def _check_value(value: Any) -> float:
    try:
        # For convenience, we allow users to report a value that can be cast to `float`.
        value = float(value)
    except (TypeError, ValueError):
        message = "The `value` argument is of type '{}' but supposed to be a float.".format(type(value).__name__)
        raise TypeError(message) from None

    return value
