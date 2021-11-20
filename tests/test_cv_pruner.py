# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import numpy as np

from cv_pruner import Method, no_features_selected, should_prune_against_threshold


def test_check_against_threshold_false():
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

    result = should_prune_against_threshold(
        current_step_of_complete_nested_cross_validation=8,
        folds_outer_cv=13,
        folds_inner_cv=8,
        validation_metric_history=validation_metric_history,
        threshold_for_pruning=0.65,
        direction_to_optimize_is_minimize=True,
        optimal_metric=0,
        method=Method.OPTIMAL_METRIC,
    )

    assert not result


def test_check_against_threshold_true():

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

    result = should_prune_against_threshold(
        current_step_of_complete_nested_cross_validation=8,
        folds_outer_cv=13,
        folds_inner_cv=8,
        validation_metric_history=validation_metric_history,
        threshold_for_pruning=0.65,
        direction_to_optimize_is_minimize=True,
        optimal_metric=0,
        method=Method.OPTIMAL_METRIC,
    )

    assert result


def test_standard_cross_validation_true():
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

    result = should_prune_against_threshold(
        current_step_of_complete_nested_cross_validation=8,
        folds_outer_cv=0,
        folds_inner_cv=8,
        validation_metric_history=validation_metric_history,
        threshold_for_pruning=0.65,
        direction_to_optimize_is_minimize=True,
        optimal_metric=0,
        method=Method.OPTIMAL_METRIC,
    )

    assert result


def test_check_no_features_selected_ndarray_false_negative():
    feature_importances = np.array([0.0, 0.0, -0.1])
    assert not no_features_selected(feature_importances)


def test_check_no_features_selected_ndarray_false():
    feature_importances = np.array([0.0, 0.0, 0.1])
    assert not no_features_selected(feature_importances)


def test_check_no_features_selected_ndarray_true():
    feature_importances = np.array([0.0, 0.0, 0.0])
    assert no_features_selected(feature_importances)


def test_check_no_features_selected_list_false_negative():
    feature_importances = [0.0, 0.0, -0.1]
    assert not no_features_selected(feature_importances)


def test_check_no_features_selected_list_true():
    feature_importances = [0.0, 0.0, 0.0]
    assert no_features_selected(feature_importances)


def test_check_no_features_selected_list_false():
    feature_importances = [0.0, 0.0, 0.1]
    assert not no_features_selected(feature_importances)
