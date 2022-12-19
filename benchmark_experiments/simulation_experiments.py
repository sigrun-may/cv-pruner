# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""TODO: add docstring."""
import datetime

import numpy as np
import pandas as pd

import benchmark_combined_pruner
from data_loader import data_loader


def colon_experiment():
    data, label = data_loader.standardize_sample_size(*data_loader.load_colon_data())

    # import optuna
    # summaries = optuna.get_all_study_summaries(storage="sqlite:///optuna_paper_db.db")
    # for summary in summaries:
    #     if "prostate_cv_pruner" in summary.study_name:
    #         optuna.study.delete_study(summary.study_name, storage="sqlite:///optuna_paper_db.db")

    for i in range(2):
        start_time = datetime.datetime.now()
        study_name = f"colon_cv_pruner_warmup4_asha_55{i}"
        print(
            "best value for metric, parameters",
            # benchmark_combined_pruner.main(data, label, study_name, threshold=0.37),
            benchmark_combined_pruner.main(data, label, study_name, threshold=0.55),
        )
        stop_time = datetime.datetime.now()
        print("duration colon:", stop_time - start_time)


def prostate_experiment():
    data, label = data_loader.standardize_sample_size(*data_loader.load_prostate_data())

    for i in range(2):
        start_time = datetime.datetime.now()
        study_name = f"prostate_cv_pruner_{i}"
        print(
            "best value for metric, parameters",
            # benchmark_combined_pruner.main(data, label, study_name, threshold=0.35),
            benchmark_combined_pruner.main(data, label, study_name, threshold=0.5),
        )
        stop_time = datetime.datetime.now()
        print("duration prostate:", stop_time - start_time)


def leukemia_experiment():
    data, label = data_loader.standardize_sample_size(*data_loader.load_leukemia_data())

    for i in range(2):
        start_time = datetime.datetime.now()
        study_name = f"leukemia_cv_pruner_{i}"
        print(
            "best value for metric, parameters",
            benchmark_combined_pruner.main(data, label, study_name, threshold=0.2),
        )
        stop_time = datetime.datetime.now()
        print("duration leukemia:", stop_time - start_time)


def enlarged_leukemia_experiment():
    data, label = data_loader.standardize_sample_size(*data_loader.load_leukemia_data())

    # enlarge data set
    artificial_features2 = np.random.normal(loc=0, scale=2, size=(30, 100000 - data.shape[1]))
    enlarged_data2 = np.hstack((data.values, artificial_features2))
    data2 = pd.DataFrame(enlarged_data2)

    print(data2.shape)

    for i in range(2):
        start_time = datetime.datetime.now()
        study_name = f"enlarged_leukemia_cv_pruner_{i}"
        print(
            "best value for metric, parameters",
            benchmark_combined_pruner.main(data2, label, study_name, threshold=0.2),
        )
        stop_time = datetime.datetime.now()
        print("duration leukemia:", stop_time - start_time)


def main():
    colon_experiment()
    prostate_experiment()
    leukemia_experiment()
    enlarged_leukemia_experiment()


if __name__ == "__main__":
    main()
