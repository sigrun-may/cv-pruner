# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Pruner for hyperparameter optimization with nested cross-validation and high variance."""

import math
from enum import Enum
from statistics import mean, median
from typing import List, Union

import numpy as np


class Method(Enum):
    """The extrapolation method."""

    # No extrapolation. Prune against the current median.
    MEDIAN = 0

    # Max deviation from median in direction to optimize as basis
    # for the extrapolation of missing performance evaluation metrics to complete the inner cross-validation.
    MAX_DEVIATION_TO_MEDIAN = 1

    # Mean deviation from median in direction to optimize as basis
    # for the extrapolation of missing performance evaluation metrics to complete the inner cross-validation.
    MEAN_DEVIATION_TO_MEDIAN = 2

    # optimal value for the performance evaluation metric as basis
    # for the extrapolation of missing performance evaluation metrics to complete the inner cross-validation.
    OPTIMAL_METRIC = 3


# for maximizing pass the negative values:  validation_metric_history * (-1)
def _extrapolate_metric(validation_metric_history: List[float], method: Method) -> float:
    """Extrapolate metric for missing values of a completed inner cross-validation loop.

    Args:
        validation_metric_history: list of all previously calculated performance evaluation metrics
        method: The extrapolation method to be used.

    Returns:
        Extrapolated performance evaluation metric based on all previously calculated performance evaluation metrics.
    """
    median_validation_metric = median(validation_metric_history)

    # get deviations from median in direction to optimize
    deviation_list = [
        median_validation_metric - metric for metric in validation_metric_history if metric < median_validation_metric
    ]

    if len(deviation_list) > 0:

        if method == Method.MEAN_DEVIATION_TO_MEDIAN:
            # mean deviation in direction to optimize from median
            return median_validation_metric - mean(deviation_list)

        elif method == Method.MAX_DEVIATION_TO_MEDIAN:
            # mean deviation in direction to optimize from median
            return median_validation_metric - max(deviation_list)
        else:
            raise ValueError(f"'method' must be of type 'Method' but was: {method}")
    else:
        return median_validation_metric


def check_against_threshold(
    current_step_of_complete_nested_cross_validation: int,
    folds_outer_cv: int,
    folds_inner_cv: int,
    validation_metric_history: List[float],
    threshold_for_pruning: float,
    direction_to_optimize_is_minimize: bool,
    optimal_metric: float,
    method: Method = Method.OPTIMAL_METRIC,
) -> bool:
    """Pruner to detect invalid metrics of the trials.

    Prune if a metric exceeds upper threshold,
    falls behind lower threshold.

    Args:
        current_step_of_complete_nested_cross_validation: one based step of complete nested cross-validation.
        folds_outer_cv: absolute number of folds for the outer cross validation loop (one based):
                        set zero for standard cross-validation.
        folds_inner_cv: absolute number of folds for the inner cross validation loop (one based).
        validation_metric_history: list of all previously calculated performance evaluation metrics.
        threshold_for_pruning: threshold which must not be exceeded (minimizing) or fallen below (maximizing).
        direction_to_optimize_is_minimize: True - in case of minimizing and False - in case of maximizing.
        optimal_metric: optimal value for the performance evaluation metric.
        method: The extrapolation method to be used.

    Returns:
        If the trial should be pruned.
    """
    current_step_inner_cv = current_step_of_complete_nested_cross_validation % folds_inner_cv

    # as 0 is not defined as valid step number, step number is reset to number of inner folds
    if current_step_inner_cv == 0:
        current_step_inner_cv = folds_inner_cv
    prune = False

    first_third_of_complete_nested_cross_validation = folds_outer_cv / 3

    # in case of standard cross-validation only one "inner-loop" is calculated
    if folds_outer_cv == 0:
        first_third_of_complete_nested_cross_validation = float("inf")

    # starts calculating after half of the inner k-fold cross-validation of the
    # nested cross-validation and a minimum of four steps
    if (
        current_step_of_complete_nested_cross_validation
        >= math.floor(folds_inner_cv / 2)
        < first_third_of_complete_nested_cross_validation
        and current_step_of_complete_nested_cross_validation > 3
    ):
        # change sign to adapt the calculations to maximize as direction to optimize
        if not direction_to_optimize_is_minimize:
            validation_metric_history = [metric * -1 for metric in validation_metric_history]
            threshold_for_pruning *= -1

        if method == Method.MEDIAN:
            return median(validation_metric_history) < threshold_for_pruning

        # extrapolate metric for the rest of the inner cross validation loop
        if method == Method.OPTIMAL_METRIC:
            extrapolated_metric = optimal_metric
        else:
            extrapolated_metric = _extrapolate_metric(validation_metric_history, method)

        # extrapolate metric up to the next complete loop of the inner k-fold cross-validation  # noqa: E501
        result_already_calculated_steps = (
            median(validation_metric_history) * current_step_of_complete_nested_cross_validation
        )
        total_steps_up_to_the_next_completed_inner_cv_loop = current_step_of_complete_nested_cross_validation + (
            folds_inner_cv - current_step_inner_cv
        )
        extrapolated_result_remaining_steps_of_the_inner_cv_loop = extrapolated_metric * (
            folds_inner_cv - current_step_inner_cv
        )
        extrapolated_result_for_next_complete_inner_cv_loop = (
            result_already_calculated_steps + extrapolated_result_remaining_steps_of_the_inner_cv_loop
        ) / total_steps_up_to_the_next_completed_inner_cv_loop

        # extrapolated results worse than threshold?
        prune = threshold_for_pruning < extrapolated_result_for_next_complete_inner_cv_loop

    return prune


def check_no_features_selected(feature_importances: Union[np.ndarray, List[float]]) -> bool:
    """Pruner to detect semantically meaningless trials.

    Prune if a trial includes a training result without any selected features.

    Args:
        feature_importances: weights or importances for each feature after training

    Returns:
        If a trial should be pruned.
    """
    if isinstance(feature_importances, list):
        feature_importances = np.array(feature_importances)

    return np.sum(feature_importances != 0) == 0
