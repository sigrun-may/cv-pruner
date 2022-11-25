# -*- coding: utf-8 -*-
# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# Copyright (c) 2020 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data loader module."""

from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def load_colon_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load colon data.

    The data is loaded and parsed from the internet.
    Also see <http://genomics-pubs.princeton.edu/oncology/affydata/index.html>

    Returns:
        Tuple containing labels and data.
    """
    cache_data_file = "./colon_data.pkl.gz"
    try:
        data_df, label = joblib.load(cache_data_file)
    except:
        html_data = "http://genomics-pubs.princeton.edu/oncology/affydata/I2000.html"

        page = requests.get(html_data)

        soup = BeautifulSoup(page.content, "html.parser")
        colon_text_data = soup.get_text()

        colon_text_data_lines = colon_text_data.splitlines()
        colon_text_data_lines = [[float(s) for s in line.split()] for line in colon_text_data_lines if len(line) > 20]
        assert len(colon_text_data_lines) == 2000
        assert len(colon_text_data_lines[0]) == 62

        data = np.array(colon_text_data_lines).T

        html_label = "http://genomics-pubs.princeton.edu/oncology/affydata/tissues.html"
        page = requests.get(html_label)
        soup = BeautifulSoup(page.content, "html.parser")
        colon_text_label = soup.get_text()
        colon_text_label = colon_text_label.splitlines()

        label = []

        for line in colon_text_label:
            try:
                i = int(line)
                label.append(0 if i > 0 else 1)
            except:  # noqa: E722
                pass

        assert len(label) == 62

        data_df = pd.DataFrame(data)

        # generate feature names
        column_names = []
        for column_name in data_df.columns:
            column_names.append("gene_" + str(column_name))

        data_df.columns = column_names

        # cache data
        joblib.dump((data_df, label), cache_data_file, compress=("gzip", 3))

    return data_df, pd.Series(label)


def load_prostate_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load prostate data.

    The data is loaded and parsed from <https://web.stanford.edu/~hastie/CASI_files/DATA/prostate.html>

    Returns:
        Tuple containing labels and data.
    """
    cache_data_file = "./prostate_data.pkl.gz"
    try:
        data, labels = joblib.load(cache_data_file)
    except:
        df = pd.read_csv("https://web.stanford.edu/~hastie/CASI_files/DATA/prostmat.csv")
        data = df.T

        # labels
        labels = []
        for label in df.columns:  # pylint:disable=no-member
            if "control" in label:
                labels.append(0)
            elif "cancer" in label:
                labels.append(1)
            else:
                assert False, "This must not happen!"

        assert len(labels) == 102
        assert data.shape == (102, 6033)

        # cache data
        joblib.dump((data, labels), cache_data_file, compress=("gzip", 3))

    return data, pd.Series(labels)


def load_leukemia_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load leukemia data.

    The data is loaded and parsed from the internet.
    Also see <https://web.stanford.edu/~hastie/CASI_files/DATA/leukemia.html>

    Returns:
        Tuple containing labels and data.
    """
    cache_data_file = "./leukemia_data.pkl.gz"
    try:
        data, labels = joblib.load(cache_data_file)
    except:
        df = pd.read_csv("https://web.stanford.edu/~hastie/CASI_files/DATA/leukemia_big.csv")
        data = df.T

        # labels
        labels = []
        for label in df.columns:  # pylint:disable=no-member
            if "ALL" in label:
                labels.append(0)
            elif "AML" in label:
                labels.append(1)
            else:
                assert False, "This must not happen!"

        assert len(labels) == 72
        assert data.shape == (72, 7128)

        # cache data
        joblib.dump((data, labels), cache_data_file, compress=("gzip", 3))

    return data, pd.Series(labels)


def standardize_sample_size(data, label) -> Tuple[pd.DataFrame, pd.Series]:
    """Reduce samples to 15 for each class.

    Returns:
        Tuple containing balanced data and corresponding labels.
    """
    # reduce samples to 15 for each class
    indices: List[int] = []
    for i in range(label.shape[0]):
        if label.iloc[i] == 0:
            indices.append(i)
        if len(indices) == 15:
            break

    for i in range(label.shape[0]):
        if label.iloc[i] == 1:
            indices.append(i)
        if len(indices) == 30:
            break

    data = data.iloc[indices]
    label = label.iloc[indices]
    assert data.shape[0] == 30
    assert len(label) == 30

    print(data.shape)
    print(label.shape)

    return data, label
