# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import numpy as np

from cv_pruner import Method, check_against_threshold, check_no_features_selected


def test_check_against_threshold_false():
    current_step_of_complete_nested_cross_validation = 8
    all_steps = 100
    folds_inner_cv = 8
    validation_metric_history = [
        0.5462845376904258,
        1.0892814295723676,
        0.6071417757721475,
        0.8962837776591452,
        0.4806071316499426,
        0.8345930756271068,
        0.6275179297604803,
        0.5881211822461981,
    ]
    threshold_for_pruning = 0.65
    direction_to_optimize_is_minimize = True
    optimal_metric = 0
    method = Method.OPTIMAL_METRIC

    result = check_against_threshold(
        current_step_of_complete_nested_cross_validation=current_step_of_complete_nested_cross_validation,
        all_steps=all_steps,
        folds_inner_cv=folds_inner_cv,
        validation_metric_history=validation_metric_history,
        threshold_for_pruning=threshold_for_pruning,
        direction_to_optimize_is_minimize=direction_to_optimize_is_minimize,
        optimal_metric=optimal_metric,
        method=method,
    )

    assert not result


def test_check_against_threshold_true():
    current_step_of_complete_nested_cross_validation = 8
    all_steps = 100
    folds_inner_cv = 8
    validation_metric_history = [
        0.6462845376904258,
        1.0892814295723676,
        0.6071417757721475,
        0.8962837776591452,
        0.6806071316499426,
        0.8345930756271068,
        0.6275179297604803,
        0.5881211822461981,
    ]
    threshold_for_pruning = 0.65
    direction_to_optimize_is_minimize = True
    optimal_metric = 0
    method = Method.OPTIMAL_METRIC

    result = check_against_threshold(
        current_step_of_complete_nested_cross_validation=current_step_of_complete_nested_cross_validation,
        all_steps=all_steps,
        folds_inner_cv=folds_inner_cv,
        validation_metric_history=validation_metric_history,
        threshold_for_pruning=threshold_for_pruning,
        direction_to_optimize_is_minimize=direction_to_optimize_is_minimize,
        optimal_metric=optimal_metric,
        method=method,
    )

    assert result


def test_check_no_features_selected_ndarray_false_negative():
    feature_importances = np.array([0.0, 0.0, -0.1])
    no_features_selected = check_no_features_selected(feature_importances)

    assert not no_features_selected


def test_check_no_features_selected_ndarray_false():
    feature_importances = np.array([0.0, 0.0, 0.1])
    no_features_selected = check_no_features_selected(feature_importances)

    assert not no_features_selected

def test_check_no_features_selected_ndarray_true():
    feature_importances = np.array([0.0, 0.0, 0.0])
    no_features_selected = check_no_features_selected(feature_importances)

    assert no_features_selected


def test_check_no_features_selected_list_false_negative():
    feature_importances = [0.0, 0.0, -0.1]
    no_features_selected = check_no_features_selected(feature_importances)

    assert not no_features_selected


def test_check_no_features_selected_list_true():
    feature_importances = [0.0, 0.0, 0.0]
    no_features_selected = check_no_features_selected(feature_importances)

    assert no_features_selected


def test_check_no_features_selected_list_false():
    feature_importances = [0.0, 0.0, 0.1]
    no_features_selected = check_no_features_selected(feature_importances)

    assert not no_features_selected
