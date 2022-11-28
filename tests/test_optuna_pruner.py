# Copyright (c) 2022 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from unittest.mock import MagicMock

from cv_pruner.optuna_pruner import MultiPrunerDelegate


def test_MultiPrunerDelegate_prune_false():
    pruner_1 = MagicMock()
    pruner_1.prune.return_value = False

    pruner_2 = MagicMock()
    pruner_2.prune.return_value = False

    pruner_list = [pruner_1, pruner_2]

    mpd = MultiPrunerDelegate(pruner_list=pruner_list)

    prune_result = mpd.prune(None, None)
    assert prune_result is False
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
