# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""TODO: add docstring."""
import datetime

import benchmark_combined_pruner
from data_loader import data_loader


def colon_experiment():
    data, label = data_loader.standardize_sample_size(*data_loader.load_colon_data())

    # import optuna
    # summaries = optuna.get_all_study_summaries(storage="sqlite:///optuna_paper_db.db")
    # for summary in summaries:
    #     if "prostate_cv_pruner" in summary.study_name:
    #         optuna.study.delete_study(summary.study_name, storage="sqlite:///optuna_paper_db.db")

    for i in range(30):
        start_time = datetime.datetime.now()
        study_name = f"colon_cv_pruner_{i}"
        print(
            "best value for metric, parameters",
            benchmark_combined_pruner.main(data, label, study_name),
        )
        stop_time = datetime.datetime.now()
        print("duration colon:", stop_time - start_time)


def prostate_experiment():
    data, label = data_loader.standardize_sample_size(*data_loader.load_prostate_data())

    # import optuna
    # summaries = optuna.get_all_study_summaries(storage="sqlite:///optuna_paper_db.db")
    # for summary in summaries:
    #     if "prostate_cv_pruner" in summary.study_name:
    #         optuna.study.delete_study(summary.study_name, storage="sqlite:///optuna_paper_db.db")

    for i in range(30):
        start_time = datetime.datetime.now()
        study_name = f"prostate_cv_pruner_{i}"
        print(
            "best value for metric, parameters",
            benchmark_combined_pruner.main(data, label, study_name),
        )
        stop_time = datetime.datetime.now()
        print("duration colon:", stop_time - start_time)


def leukemia_experiment():
    data, label = data_loader.standardize_sample_size(*data_loader.load_leukemia_data())

    # import optuna
    # summaries = optuna.get_all_study_summaries(storage="sqlite:///optuna_paper_db.db")
    # for summary in summaries:
    #     if "prostate_cv_pruner" in summary.study_name:
    #         optuna.study.delete_study(summary.study_name, storage="sqlite:///optuna_paper_db.db")

    for i in range(30):
        start_time = datetime.datetime.now()
        study_name = f"leukemia_cv_pruner_{i}"
        print(
            "best value for metric, parameters",
            benchmark_combined_pruner.main(data, label, study_name),
        )
        stop_time = datetime.datetime.now()
        print("duration colon:", stop_time - start_time)


def main():
    colon_experiment()
    prostate_experiment()
    leukemia_experiment()


if __name__ == "__main__":
    main()
