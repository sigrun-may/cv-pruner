# Copyright (c) 2022 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cv_pruner.optuna_pruner import BenchmarkPruneFunctionWrapper, MultiPrunerDelegate, NoFeatureSelectedPruner


def test_MultiPrunerDelegate_prune_false():
    pruner_1 = MagicMock()
    pruner_1.prune.return_value = False

    pruner_2 = MagicMock()
    pruner_2.prune.return_value = False

    pruner_list = [pruner_1, pruner_2]

    mpd = MultiPrunerDelegate(pruner_list=pruner_list)

    prune_result = mpd.prune(None, None)
    assert not prune_result
    pruner_1.prune.assert_called_once()
    pruner_2.prune.assert_called_once()


def test_MultiPrunerDelegate_prune_true():
    pruner_1 = MagicMock()
    pruner_1.prune.return_value = False

    pruner_2 = MagicMock()
    pruner_2.prune.return_value = True

    pruner_list = [pruner_1, pruner_2]

    mpd = MultiPrunerDelegate(pruner_list=pruner_list)

    prune_result = mpd.prune(None, None)
    assert prune_result
    pruner_1.prune.assert_called_once()
    pruner_2.prune.assert_called_once()


def test_MultiPrunerDelegate_prune_true_eager():
    pruner_1 = MagicMock()
    pruner_1.prune.return_value = True

    pruner_2 = MagicMock()
    pruner_2.prune.return_value = False

    pruner_list = [pruner_1, pruner_2]

    mpd = MultiPrunerDelegate(pruner_list=pruner_list)

    prune_result = mpd.prune(None, None)
    assert prune_result
    pruner_1.prune.assert_called_once()
    pruner_2.prune.assert_not_called()


def test_MultiPrunerDelegate_prune_true_not_eager():
    pruner_1 = MagicMock()
    pruner_1.prune.return_value = True

    pruner_2 = MagicMock()
    pruner_2.prune.return_value = False

    pruner_list = [pruner_1, pruner_2]

    mpd = MultiPrunerDelegate(pruner_list=pruner_list, prune_eager=False)

    prune_result = mpd.prune(None, None)
    assert prune_result
    pruner_1.prune.assert_called_once()
    pruner_2.prune.assert_called_once()


def test_NoFeatureSelectedPruner_all_zero_list():
    feature_values = [0.0, 0.0]
    nfsp = NoFeatureSelectedPruner()
    nfsp.communicate_feature_values(feature_values)

    prune_result = nfsp.prune(None, None)
    assert prune_result


def test_NoFeatureSelectedPruner_all_zero_np():
    feature_values = np.array([0.0, 0.0])
    nfsp = NoFeatureSelectedPruner()
    nfsp.communicate_feature_values(feature_values)

    prune_result = nfsp.prune(None, None)
    assert prune_result


def test_NoFeatureSelectedPruner_not_all_zero_list():
    feature_values = [0.0, 0.1]
    nfsp = NoFeatureSelectedPruner()
    nfsp.communicate_feature_values(feature_values)

    prune_result = nfsp.prune(None, None)
    assert not prune_result


def test_NoFeatureSelectedPruner_not_all_zero_np():
    feature_values = np.array([0.0, 0.1])
    nfsp = NoFeatureSelectedPruner()
    nfsp.communicate_feature_values(feature_values)

    prune_result = nfsp.prune(None, None)
    assert not prune_result


def test_NoFeatureSelectedPruner_feature_values_not_set():
    nfsp = NoFeatureSelectedPruner()
    with pytest.raises(RuntimeError):
        nfsp.prune(None, None)


@patch("cv_pruner.optuna_pruner.datetime")
def test_BenchmarkPruneFunctionWrapper(datetime):
    fake_timestamp = "fake_timestamp"
    datetime.datetime.now.return_value = fake_timestamp

    mock_pruner = MagicMock()
    mock_pruner.prune.side_effect = [False, True, True]
    mock_pruner_name = "FakeMockPruner"
    bpfw = BenchmarkPruneFunctionWrapper(mock_pruner, pruner_name=mock_pruner_name)

    mock_trial = MagicMock()
    mock_trial.intermediate_values.values.return_value = [0.1, 0.2]

    first_prune_result = bpfw.prune(None, mock_trial)
    assert not first_prune_result

    second_prune_result = bpfw.prune(None, mock_trial)
    assert second_prune_result
    mock_trial.set_user_attr.assert_called_with(
        f"{mock_pruner_name}_pruned_at", {"step": 2, "timestamp": fake_timestamp}
    )
    mock_trial.set_user_attr.called_once()

    third_prune_result = bpfw.prune(None, mock_trial)
    assert third_prune_result
    mock_trial.set_user_attr.called_once()  # not 2 times!!!
