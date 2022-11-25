# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""TODO: add docstring."""

from datetime import timedelta
from timeit import default_timer as time

import hpo_pruner_simulation
from data_loader import data_loader


def colon_experiment():
    data, label = data_loader.standardize_sample_size(*data_loader.load_colon_data())

    # import optuna
    # summaries = optuna.get_all_study_summaries(storage="sqlite:///optuna_paper_db.db")
    # for summary in summaries:
    #     if "prostate_cv_pruner" in summary.study_name:
    #         optuna.study.delete_study(summary.study_name, storage="sqlite:///optuna_paper_db.db")

    for i in range(30):
        start = time()
        study_name = "colon_cv_pruner" + str(i)
        print(
            "best value for metric, parameters",
            hpo_pruner_simulation.optimize(data, label, study_name),
        )
        print("duration colon:", timedelta(seconds=time() - start))


def prostate_experiment():
    data, label = data_loader.standardize_sample_size(*data_loader.load_prostate_data())

    # import optuna
    # summaries = optuna.get_all_study_summaries(storage="sqlite:///optuna_paper_db.db")
    # for summary in summaries:
    #     if "prostate_cv_pruner" in summary.study_name:
    #         optuna.study.delete_study(summary.study_name, storage="sqlite:///optuna_paper_db.db")

    for i in range(30):
        start = time()
        study_name = "prostate_cv_pruner" + str(i)
        print(
            "best value for metric, parameters",
            hpo_pruner_simulation.optimize(data, label, study_name),
        )
        print("duration prostate:", timedelta(seconds=time() - start))


def leukemia_experiment():
    data, label = data_loader.standardize_sample_size(*data_loader.load_leukemia_data())

    # import optuna
    # summaries = optuna.get_all_study_summaries(storage="sqlite:///optuna_paper_db.db")
    # for summary in summaries:
    #     if "prostate_cv_pruner" in summary.study_name:
    #         optuna.study.delete_study(summary.study_name, storage="sqlite:///optuna_paper_db.db")

    for i in range(30):
        start = time()
        study_name = "leukemia_cv_pruner" + str(i)
        print(
            "best value for metric, parameters",
            hpo_pruner_simulation.optimize(data, label, study_name),
        )
        print("duration leukemia:", timedelta(seconds=time() - start))


def main():
    colon_experiment()
    prostate_experiment()
    leukemia_experiment()


if __name__ == "__main__":
    main()
