# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Simulate pruning of hyperparameter optimization."""

import logging
import math
import time
import warnings
from statistics import mean, median
from typing import Dict

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna import TrialPruned
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from scipy.stats import trim_mean
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from tqdm import tqdm

import cv_pruner
import settings

warnings.filterwarnings("ignore")
logging.basicConfig(format="%(levelname)s:%(message)s")
logger = logging.getLogger("my_logger")
logger.setLevel(settings.logging_level)

DB = "sqlite:///optuna_paper_db.db"


def simulate_no_features_selected_pruning(trial, model, current_step_complete_ncv, start_time):
    """TODO: add docstring. Rename no importances"""

    # simulate pruning
    pruned = cv_pruner.no_features_selected(model.feature_importance(importance_type="gain"))
    if pruned:
        trial.set_user_attr("time_until_pruning_no_features_selected", time.time() - start_time)
        trial.set_user_attr("would_prune_no_features_selected", current_step_complete_ncv)
        logger.debug(
            "\ntrial %s has no selected features, pruned at step: %s",
            trial.number,
            current_step_complete_ncv,
        )
    return pruned


def simulate_standard_pruning(trial, step, evaluation_metric_list, start_time):
    """TODO: add docstring."""
    # built in optuna standard pruner
    standard_pruned = trial.should_prune()
    if standard_pruned:
        trial.set_user_attr("time_until_optuna_pruned", time.time() - start_time)
        trial.set_user_attr("optuna_pruned", step)
        logger.debug(
            "optuna pruned at step %s in trial %s with median %s and mean %s",
            step,
            trial.number,
            median(evaluation_metric_list),
            mean(evaluation_metric_list),
        )
    return standard_pruned


#####################################################################
# Simulate pruning of embedded feature selection
#####################################################################
def simulate_hpo_pruning(
        params: Dict, label: pd.Series, unlabeled_data: pd.DataFrame, trial: optuna.Trial
) -> float:  # pylint: disable=too-many-locals
    """TODO: add docstring: Was wird gemacht? Welche Parameter? Rückgabewert?."""
    # initialize variables
    evaluation_metric_list = []
    number_of_features_in_model_list = []
    step_duration_list = []
    current_step_complete_ncv = 0

    pruned = False
    standard_pruned = False

    start_time = time.time()
    number_of_steps = unlabeled_data.shape[0] * settings.folds_inner_cv

    # status bar for hyperparameter optimization
    if settings.show_progress:
        progress_bar = tqdm(total=number_of_steps, postfix="mean evaluation_metric: ?")

    # outer loop of the nested cross-validation
    loo = LeaveOneOut()
    for outer_sample_index, (remain_index, test_index) in enumerate(  # pylint: disable=unused-variable  # noqa: E501
            loo.split(unlabeled_data)
    ):
        x_remain = unlabeled_data.iloc[remain_index, :]
        y_remain = label.iloc[remain_index]

        # test set is not considered for the simulation of pruning
        # x_test = unlabeled_data.iloc[test_index, :]
        # y_test = label.iloc[test_index]

        assert unlabeled_data.shape[0] - 1 == x_remain.shape[0]
        assert len(label) - 1 == len(y_remain)

        # inner loop of the nested cross-validation
        folds_inner_cv = StratifiedKFold(settings.folds_inner_cv, shuffle=True, random_state=42)
        for train_index, validation_index in folds_inner_cv.split(x_remain, y_remain):

            # counting steps starts with one
            current_step_complete_ncv += 1
            start_step = time.time()

            x_train = x_remain.iloc[train_index, :]
            x_validation = x_remain.iloc[validation_index, :]
            y_train = y_remain.iloc[train_index]
            y_validation = y_remain.iloc[validation_index]

            train_data = lgb.Dataset(x_train, label=y_train)
            validation_data = lgb.Dataset(x_validation, label=y_validation)

            model = lgb.train(
                params,
                train_data,
                valid_sets=[validation_data],
                verbose_eval=False,
            )

            validation_metric = model.best_score["valid_0"]["binary_logloss"]
            evaluation_metric_list.append(validation_metric)

            if settings.show_progress:
                progress_bar.postfix = "mean evaluation_metric: " + str(mean(evaluation_metric_list))
                progress_bar.update(1)

            # save number of features in model to analyze experiments
            number_of_features_in_model_list.append(int(np.sum(model.feature_importance(importance_type="gain") > 0)))

            # simulate pruning
            if not pruned:  # check if the simulated pruner has already pruned the trial
                pruned = simulate_no_features_selected_pruning(trial, model, current_step_complete_ncv, start_time)

            # report each validation result to show the high variance
            if not settings.reduce_variance:
                trial.report(validation_metric, current_step_complete_ncv)

                # built in optuna standard pruner
                if not standard_pruned:  # check if the simulated pruner has already pruned the trial
                    standard_pruned = simulate_standard_pruning(
                        trial, current_step_complete_ncv, evaluation_metric_list, start_time
                    )

            step_duration_list.append(time.time() - start_step)

        # only report 20 % trimmed mean values of complete inner folds to smooth high variance in validation results  # noqa: E501
        if settings.reduce_variance:
            if settings.intermediate_value == "trim_mean":
                trial.report(
                    trim_mean(evaluation_metric_list, proportiontocut=0.2),
                    outer_sample_index,
                )
            elif settings.intermediate_value == "median":
                trial.report(
                    np.median(evaluation_metric_list),
                    outer_sample_index,
                )
            else:
                raise ValueError("no valid intermediate value method defined")

            # built in optuna standard pruner
            if not standard_pruned:  # check if the simulated pruner has already pruned the trial
                standard_pruned = simulate_standard_pruning(
                    trial, outer_sample_index, evaluation_metric_list, start_time
                )

            # check if pruning is only simulated
            if not settings.simulate_standard_pruner and standard_pruned:
                if settings.show_progress:
                    progress_bar.close()
                raise TrialPruned()

    logger.debug(
        "\ntrial %s: mean %s - median %s - trimmed mean %s",
        trial.number,
        mean(evaluation_metric_list),
        median(evaluation_metric_list),
        trim_mean(evaluation_metric_list, proportiontocut=0.2),
    )

    if settings.show_progress:
        progress_bar.close()

    # save raw metric to analyze experiments
    trial.set_user_attr("raw_evaluation_metric_list", evaluation_metric_list)
    trial.set_user_attr("number_of_features_in_model_list", number_of_features_in_model_list)
    trial.set_user_attr("step_duration_list", step_duration_list)

    # only report 20 % trimmed mean values of complete inner folds to smooth high variance in validation results  # noqa: E501
    if settings.reduce_variance:
        if settings.intermediate_value == "trim_mean":
            # return trimmed mean to the optimizer to prevent optimizing based on outliers
            return trim_mean(evaluation_metric_list, proportiontocut=0.2)
        elif settings.intermediate_value == "median":
            return np.median(evaluation_metric_list)
        else:
            raise ValueError("no valid intermediate value method defined")


def optimize(data, label, study_name):
    """TODO: add docstring."""

    def optuna_objective(trial):
        sample_size = data.shape[0]

        parameters = dict(
            # regularization:
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 2, math.floor(sample_size / 2)),
            lambda_l1=trial.suggest_uniform("lambda_l1", 0.0, 3),
            min_gain_to_split=trial.suggest_uniform("min_gain_to_split", 0, 5),
            # necessary for random forest:
            boosting_type=settings.boosting_type,
            max_depth=trial.suggest_int("max_depth", 2, 15),
            bagging_fraction=trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
            num_boost_round=100,
            # settings for lightgbm
            objective="binary",
            metric=settings.metric,
            verbose=-1,
            # parallelization on the level of trials is more effective
            n_jobs=1,
        )

        # num_leaves must be smaller than 2^max_depth: 2^max_depth > num_leaves
        max_num_leaves = (2 ** parameters["max_depth"]) - 1

        # limit num_leaves
        max_num_leaves = min(max_num_leaves, 80)
        parameters["num_leaves"] = trial.suggest_int("num_leaves", 2, max_num_leaves)

        assert (2 ** parameters["max_depth"]) > parameters["num_leaves"], (
                'Parameter failure -> parameters["max_depth"] > parameters["num_leaves"]:'
                + str(parameters["max_depth"])
                + ">"
                + str(parameters["num_leaves"])
        )
        return simulate_hpo_pruning(parameters, label, data, trial)

    # optuna.study.delete_study(study_name, storage=DB)
    study = optuna.create_study(
        storage=DB,
        study_name=study_name,
        load_if_exists=True,
        direction=settings.direction_to_optimize,
        sampler=TPESampler(multivariate=True),
        pruner=SuccessiveHalvingPruner(
            min_resource="auto",
            reduction_factor=3,
            min_early_stopping_rate=2,
            bootstrap_count=0,
        ),
    )

    start_time = time.time()

    study.optimize(
        optuna_objective,
        n_trials=settings.n_trials,
        n_jobs=settings.n_jobs_optuna,
    )

    return study.best_value, study.best_params, start_time - time.time()
