# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Pruner for hyperparameter optimization with nested cross-validation and high variance."""

from enum import Enum
from statistics import mean, median
from typing import List, Optional, Union

import numpy as np


class Method(Enum):
    """Extrapolation method for the threshold-based pruner.

    MEDIAN
        No extrapolation. Pruning against the current median.
    MAX_DEVIATION_TO_MEDIAN
        Maximum deviation from the median in direction to optimize
        serves as basis for the extrapolation of missing performance
        evaluation values of the complete inner cross-validation.
    MEAN_DEVIATION_TO_MEDIAN
        Mean deviation from the median in direction to optimize
        serves as basis for the extrapolation of missing performance
        evaluation values of the complete inner cross-validation.
    OPTIMAL_METRIC
        Optimal value for the performance evaluation metric serves as basis
        for the extrapolation of missing performance
        evaluation values of the complete inner cross-validation.
    """

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


def should_prune_against_threshold(
        folds_inner_cv: int,
        validation_metric_history: List[float],
        threshold_for_pruning: float,
        start_step: int,
        stop_step: int,
        direction_to_optimize_is_minimize: bool,
        method: Method = Method.MEAN_DEVIATION_TO_MEDIAN,
        optimal_metric: Optional[float] = None,
) -> bool:
    """Pruner to detect an invalid performance evaluation value of a trial.

    Args:
        folds_inner_cv: Absolute number of folds for the inner cross
            validation loop (one based).
        validation_metric_history: List of all previously calculated performance evaluation metric values.
        threshold_for_pruning: Threshold that should not be exceeded
            (minimizing) or fallen below (maximizing).
        start_step: Pruning starts after a patience of the specified ``start_step`` steps. Must be greater than 2.
            It is recommended to choose ``start_step`` greater than 3.
        stop_step: Pruning stops after the specified ``stop_step`` steps. Must be greater than ``start_step``.
        direction_to_optimize_is_minimize: True - in case of minimizing and False - in case of maximizing.
        method: The extrapolation method to be used (see Method).
        optimal_metric: Optimal value for the performance evaluation metric.
            Must be set if extrapolation method is ``Metric.OPTIMAL_METRIC``.

    Returns:
        If the trial should be pruned. TRUE if it is likely that the final
        performance evaluation metric will exceed the upper threshold or
        fall below the lower threshold respectively.
        FALSE otherwise.
    """
    if not start_step > 2:
        raise ValueError("start_step must be greater than 2!")
    if not stop_step > start_step:
        raise ValueError("stop_step must be greater than start_step!")
    if method == Method.OPTIMAL_METRIC and optimal_metric is None:
        raise ValueError("optimal_metric must be set if extrapolation method is Metric.OPTIMAL_METRIC!")

    prune = False
    current_step_of_complete_nested_cross_validation = len(
        validation_metric_history
    )  # number of all steps s calculated so far

    # consider range for pruning
    if start_step < current_step_of_complete_nested_cross_validation <= stop_step:

        # change sign to adapt the calculations to maximize as direction to optimize
        if not direction_to_optimize_is_minimize:
            validation_metric_history = [metric * -1 for metric in validation_metric_history]
            threshold_for_pruning *= -1

        if method == Method.MEDIAN:
            return threshold_for_pruning < median(validation_metric_history)

        # extrapolate metric for the rest of the inner cross validation loop
        elif method == Method.OPTIMAL_METRIC:
            extrapolated_metric = optimal_metric  # optimistically extrapolated value e
        else:
            extrapolated_metric = _extrapolate_metric(
                validation_metric_history, method
            )  # optimistically extrapolated value e

        # extrapolate metric up to the next complete loop of the inner k-fold cross-validation  # noqa: E501
        result_already_calculated_steps = (
                median(validation_metric_history) * current_step_of_complete_nested_cross_validation
        )

        #  extrapolate remaining steps of current inner cross-validation loop
        #  no extrapolation if inner cross-validation loop is complete

        # calculate number of steps up to the next completed inner cross-validation loop
        current_step_inner_cv = current_step_of_complete_nested_cross_validation % folds_inner_cv
        if current_step_inner_cv == 0:  # inner cross-validation loop is complete
            remaining_steps_of_inner_cross_validation_loop = (
                0  # no missing steps m until the next complete inner cross-validation loop
            )
        else:
            remaining_steps_of_inner_cross_validation_loop = (
                    folds_inner_cv - current_step_inner_cv
            )  # number of missing steps m until the next complete inner cross-validation loop

        total_steps_up_to_the_next_completed_inner_cv_loop = (
                current_step_of_complete_nested_cross_validation + remaining_steps_of_inner_cross_validation_loop
        )
        extrapolated_result_of_remaining_steps_of_the_inner_cv_loop = (
                extrapolated_metric * remaining_steps_of_inner_cross_validation_loop  # type: ignore
        )
        extrapolated_result_for_next_complete_inner_cv_loop = (
                                                                      result_already_calculated_steps + extrapolated_result_of_remaining_steps_of_the_inner_cv_loop
                                                              ) / total_steps_up_to_the_next_completed_inner_cv_loop

        # extrapolated results worse than threshold?
        prune = threshold_for_pruning < extrapolated_result_for_next_complete_inner_cv_loop

    return prune


def no_features_selected(feature_importances: Union[np.ndarray, List[float]]) -> bool:
    """Pruner to detect semantically meaningless trials.

    Args:
        feature_importances: Weights, importances or coefficients for each
            feature after training.

    Returns:
        If a trial should be pruned. TRUE if a trial includes a training
        result without any selected features. FALSE otherwise.

    """
    if isinstance(feature_importances, list):
        feature_importances = np.array(feature_importances)

    return np.sum(feature_importances != 0) == 0
