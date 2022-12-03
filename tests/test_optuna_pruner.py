# Copyright (c) 2022 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from unittest.mock import MagicMock, Mock, PropertyMock, create_autospec, patch

import numpy as np
import pytest
from optuna import Study
from optuna.pruners import BasePruner, NopPruner
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from scipy.stats import trim_mean

from cv_pruner.optuna_pruner import (
    BenchmarkPruneFunctionWrapper,
    MultiPrunerDelegate,
    NoModelBuildPruner,
    RepeatedTrainingPrunerWrapper,
    RepeatedTrainingThresholdPruner,
)


def test_MultiPrunerDelegate_prune_false():
    pruner_1 = create_autospec(BasePruner)
    pruner_1.prune.return_value = False

    pruner_2 = create_autospec(BasePruner)
    pruner_2.prune.return_value = False

    pruner_list = [pruner_1, pruner_2]

    mpd = MultiPrunerDelegate(pruner_list=pruner_list)

    prune_result = mpd.prune(None, None)
    assert not prune_result
    pruner_1.prune.assert_called_once()
    pruner_2.prune.assert_called_once()


def test_MultiPrunerDelegate_prune_true():
    pruner_1 = create_autospec(BasePruner)
    pruner_1.prune.return_value = False

    pruner_2 = create_autospec(BasePruner)
    pruner_2.prune.return_value = True

    pruner_list = [pruner_1, pruner_2]

    mpd = MultiPrunerDelegate(pruner_list=pruner_list)

    prune_result = mpd.prune(None, None)
    assert prune_result
    pruner_1.prune.assert_called_once()
    pruner_2.prune.assert_called_once()


def test_MultiPrunerDelegate_prune_true_eager():
    pruner_1 = create_autospec(BasePruner)
    pruner_1.prune.return_value = True

    pruner_2 = create_autospec(BasePruner)
    pruner_2.prune.return_value = False

    pruner_list = [pruner_1, pruner_2]

    mpd = MultiPrunerDelegate(pruner_list=pruner_list)

    prune_result = mpd.prune(None, None)
    assert prune_result
    pruner_1.prune.assert_called_once()
    pruner_2.prune.assert_not_called()


def test_MultiPrunerDelegate_prune_true_not_eager():
    pruner_1 = create_autospec(BasePruner)
    pruner_1.prune.return_value = True

    pruner_2 = create_autospec(BasePruner)
    pruner_2.prune.return_value = False

    pruner_list = [pruner_1, pruner_2]

    mpd = MultiPrunerDelegate(pruner_list=pruner_list, prune_eager=False)

    prune_result = mpd.prune(None, None)
    assert prune_result
    pruner_1.prune.assert_called_once()
    pruner_2.prune.assert_called_once()


def test_NoFeatureSelectedPruner_all_zero_list():
    feature_values = [0.0, 0.0]
    pruner = NoModelBuildPruner()
    pruner.communicate_feature_values(feature_values)

    prune_result = pruner.prune(None, None)
    assert prune_result


def test_NoFeatureSelectedPruner_all_zero_np():
    feature_values = np.array([0.0, 0.0])
    pruner = NoModelBuildPruner()
    pruner.communicate_feature_values(feature_values)

    prune_result = pruner.prune(None, None)
    assert prune_result


def test_NoFeatureSelectedPruner_not_all_zero_list():
    feature_values = [0.0, 0.1]
    pruner = NoModelBuildPruner()
    pruner.communicate_feature_values(feature_values)

    prune_result = pruner.prune(None, None)
    assert not prune_result


def test_NoFeatureSelectedPruner_not_all_zero_np():
    feature_values = np.array([0.0, 0.1])
    pruner = NoModelBuildPruner()
    pruner.communicate_feature_values(feature_values)

    prune_result = pruner.prune(None, None)
    assert not prune_result


def test_NoFeatureSelectedPruner_feature_values_not_set():
    pruner = NoModelBuildPruner()
    with pytest.raises(RuntimeError):
        pruner.prune(None, None)


@patch("cv_pruner.optuna_pruner.datetime")
def test_BenchmarkPruneFunctionWrapper(datetime):
    fake_timestamp = "fake_timestamp"
    datetime.datetime.now.return_value.isoformat.return_value = fake_timestamp

    mock_pruner = create_autospec(NopPruner)
    mock_pruner.prune.side_effect = [False, True, True]
    pruner = BenchmarkPruneFunctionWrapper(mock_pruner)

    mock_trial = MagicMock()
    mock_trial.intermediate_values.values.return_value = [0.1, 0.2]

    study_mock = MagicMock()
    # study._storage.set_trial_user_attr(trial._trial_id, f"{self.pruner_name}_pruned_at", {"step": step, "timestamp": pruning_timestamp})
    study_mock._storage.set_trial_user_attr.return_value = None

    first_prune_result = pruner.prune(study_mock, mock_trial)
    assert not first_prune_result  # we never prune
    assert not pruner.pruned_trial_numbers

    second_prune_result = pruner.prune(study_mock, mock_trial)
    assert not second_prune_result  # we never prune
    assert len(pruner.pruned_trial_numbers) == 1
    # mock_trial.set_user_attr.assert_called_with("NopPruner_pruned_at", {"step": 2, "timestamp": fake_timestamp})
    study_mock._storage.set_trial_user_attr.assert_called_once()

    third_prune_result = pruner.prune(study_mock, mock_trial)
    assert not third_prune_result  # we never prune
    assert len(pruner.pruned_trial_numbers) == 1
    study_mock._storage.set_trial_user_attr.assert_called_once()


@patch("cv_pruner.optuna_pruner.should_prune_against_threshold")
def test_RepeatedTrainingThresholdPruner_no_prune_if_last_step_not_set(should_prune_against_threshold_mock):
    should_prune_against_threshold_mock.return_value = True

    mock_trial = create_autospec(FrozenTrial)
    none_property_mock = PropertyMock(return_value=None)
    type(mock_trial).last_step = none_property_mock
    pruner = RepeatedTrainingThresholdPruner(threshold=0.5)
    prune_result = pruner.prune(None, mock_trial)

    assert not prune_result
    none_property_mock.assert_called_once_with()
    should_prune_against_threshold_mock.assert_not_called()


@patch("cv_pruner.optuna_pruner.should_prune_against_threshold")
def test_RepeatedTrainingThresholdPruner_no_prune_if_before_n_warmup_steps(should_prune_against_threshold_mock):
    should_prune_against_threshold_mock.return_value = True

    # create trial_mock
    trial_mock = create_autospec(FrozenTrial)
    none_property_mock = PropertyMock(return_value="something")
    type(trial_mock).last_step = none_property_mock
    # trial.intermediate_values.values()
    values_mock = Mock()
    values_mock.values.return_value = range(4)
    trial_mock.intermediate_values = values_mock

    # create study_mock
    study_mock = create_autospec(Study)
    # study.direction == StudyDirection.MINIMIZE
    study_mock.direction == StudyDirection.MINIMIZE

    pruner = RepeatedTrainingThresholdPruner(threshold=0.5, n_warmup_steps=5)
    prune_result = pruner.prune(study_mock, trial_mock)

    assert not prune_result
    none_property_mock.assert_called_once_with()
    trial_mock.intermediate_values.values.assert_called_once_with()
    should_prune_against_threshold_mock.assert_not_called()


@patch("cv_pruner.optuna_pruner.should_prune_against_threshold")
def test_RepeatedTrainingThresholdPruner_no_prune_if_after_active_until_step(should_prune_against_threshold_mock):
    should_prune_against_threshold_mock.side_effect = RuntimeError("Should not execute this!")

    # create trial_mock
    trial_mock = create_autospec(FrozenTrial)
    none_property_mock = PropertyMock(return_value="something")
    type(trial_mock).last_step = none_property_mock
    # trial.intermediate_values.values()
    values_mock = Mock()
    values_mock.values.return_value = range(6)
    trial_mock.intermediate_values = values_mock

    # create study_mock
    study_mock = create_autospec(Study)
    # study.direction == StudyDirection.MINIMIZE
    study_mock.direction == StudyDirection.MINIMIZE

    pruner = RepeatedTrainingThresholdPruner(threshold=0.5, active_until_step=5)
    prune_result = pruner.prune(study_mock, trial_mock)

    assert not prune_result
    none_property_mock.assert_called_once_with()
    trial_mock.intermediate_values.values.assert_called_once_with()
    should_prune_against_threshold_mock.assert_not_called()


@patch("cv_pruner.optuna_pruner.should_prune_against_threshold")
def test_RepeatedTrainingThresholdPruner_active_between_n_warmup_steps_and_active_until_step_low_value(
        should_prune_against_threshold_mock,
):
    should_prune_against_threshold_mock.return_value = True

    # create trial_mock
    trial_mock = create_autospec(FrozenTrial)
    none_property_mock = PropertyMock(return_value="something")
    type(trial_mock).last_step = none_property_mock
    # trial.intermediate_values.values()
    values_mock = Mock()
    values_mock.values.return_value = range(5)
    trial_mock.intermediate_values = values_mock

    # create study_mock
    study_mock = create_autospec(Study)
    # study.direction == StudyDirection.MINIMIZE
    study_mock.direction == StudyDirection.MINIMIZE

    pruner = RepeatedTrainingThresholdPruner(threshold=0.5, n_warmup_steps=5, active_until_step=10)
    prune_result = pruner.prune(study_mock, trial_mock)

    assert prune_result
    none_property_mock.assert_called_once_with()
    trial_mock.intermediate_values.values.assert_called_once_with()
    should_prune_against_threshold_mock.assert_called_once()


@patch("cv_pruner.optuna_pruner.should_prune_against_threshold")
def test_RepeatedTrainingThresholdPruner_active_between_n_warmup_steps_and_active_until_step_high_value(
        should_prune_against_threshold_mock,
):
    should_prune_against_threshold_mock.return_value = True

    # create trial_mock
    trial_mock = create_autospec(FrozenTrial)
    something_property_mock = PropertyMock(return_value="something")
    type(trial_mock).last_step = something_property_mock
    # trial.intermediate_values.values()
    values_mock = Mock()
    values_mock.values.return_value = range(10)
    trial_mock.intermediate_values = values_mock

    # create study_mock
    study_mock = create_autospec(Study)
    # study.direction == StudyDirection.MINIMIZE
    study_mock.direction == StudyDirection.MINIMIZE

    pruner = RepeatedTrainingThresholdPruner(threshold=0.5, n_warmup_steps=5, active_until_step=10)
    prune_result = pruner.prune(study_mock, trial_mock)

    assert prune_result
    something_property_mock.assert_called_once_with()
    trial_mock.intermediate_values.values.assert_called_once_with()
    should_prune_against_threshold_mock.assert_called_once()


def test_RepeatedTrainingPrunerWrapper_no_intermediate_values():
    values = [0.1, 0.1, 0.2, 0.3, 0.3]
    expected_aggregated_value = trim_mean(values[:5], proportiontocut=0.2)
    intermediate_values = {k: v for k, v in enumerate(values)}

    # build trial_mock
    trial_mock = MagicMock()
    last_step_property_mock = PropertyMock(return_value=values[-1])
    type(trial_mock).last_step = last_step_property_mock
    intermediate_values_property_mock = PropertyMock(return_value=intermediate_values)
    trial_mock.intermediate_values = intermediate_values_property_mock

    # build pruner mock
    pruner_mock = create_autospec(NopPruner)
    pruner_mock.prune.return_value = True

    pruner_wrapper = RepeatedTrainingPrunerWrapper(pruner_mock, inner_cv_folds=5)
    prune_result = pruner_wrapper.prune(None, trial_mock)

    assert prune_result
    last_step_property_mock.assert_called_once_with()
    # TODO: check if pruner_mock.prune was called with correct values
    call_args = pruner_mock.prune.call_args.args
    pass
