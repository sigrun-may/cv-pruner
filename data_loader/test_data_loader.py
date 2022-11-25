# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for data loaders."""

from data_loader.data_loader import load_colon_data, load_leukemia_data, load_prostate_data


def test_load_colon_data():
    colon_data_labels, colon_data = load_colon_data()

    assert len(colon_data_labels) == 62
    assert colon_data.shape == (62, 2000)


def test_load_prostate_data():
    prostate_data_labels, prostate_data = load_prostate_data()

    assert len(prostate_data_labels) == 102
    assert prostate_data.shape == (102, 6033)


def test_load_leukemia_data():
    leukemia_data_labels, leukemia_data = load_leukemia_data()

    assert len(leukemia_data_labels) == 72
    assert leukemia_data.shape == (72, 7128)
