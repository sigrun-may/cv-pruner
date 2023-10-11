# Copyright (c) 2022 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2022 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
import datetime
import math
from functools import partial

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import PercentilePruner, SuccessiveHalvingPruner
from scipy.stats import trim_mean
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from cv_pruner import Method
from cv_pruner.optuna_pruner import (
    BenchmarkPruneFunctionWrapper,
    MultiPrunerDelegate,
    NoModelBuildPruner,
    RepeatedTrainingPrunerWrapper,
    RepeatedTrainingThresholdPruner,
)
from data_loader.data_loader import load_colon_data


"""Accelerating embedded feature selection with combined pruning."""

# Parse the data
# df = pd.read_csv("/home/sigrun/PycharmProjects/reverse_feature_selection/data/small_50.csv")
# label = df["label"]
# data = df.iloc[:, 1:]
data, label = load_colon_data()
# data = data1.iloc[:, :40]
inner_folds: int = 10


def combined_cv_pruner_factory(inner_cv_folds, threshold, extrapolation_method, comparison_based_pruner, base_pruner):
    no_model_build_pruner = NoModelBuildPruner()
    no_model_build_pruner_benchmark = BenchmarkPruneFunctionWrapper(no_model_build_pruner)

    threshold_pruner = RepeatedTrainingThresholdPruner(
        threshold=threshold,
        extrapolation_interval=inner_cv_folds,
        extrapolation_method=extrapolation_method,
    )
    threshold_pruner = BenchmarkPruneFunctionWrapper(threshold_pruner)

    comparison_based_pruner = RepeatedTrainingPrunerWrapper(
        pruner=comparison_based_pruner, inner_cv_folds=inner_cv_folds
    )  # no not switch order with BenchmarkPruneFunctionWrapper!
    comparison_based_pruner = BenchmarkPruneFunctionWrapper(
        comparison_based_pruner, pruner_name="comparison"
    )  # no not switch order with RepeatedTrainingPrunerWrapper!

    base_pruner = RepeatedTrainingPrunerWrapper(
        pruner=base_pruner, inner_cv_folds=1
    )  # no not switch order with BenchmarkPruneFunctionWrapper!
    base_pruner = BenchmarkPruneFunctionWrapper(
        base_pruner, pruner_name="comparison_baseline"
    )  # no not switch order with RepeatedTrainingPrunerWrapper!

    compound_pruner = MultiPrunerDelegate(
        # pruner_list=[no_model_build_pruner_benchmark, threshold_pruner, comparison_based_pruner], prune_eager=False
        pruner_list=[comparison_based_pruner],
        prune_eager=False,
    )

    return compound_pruner, no_model_build_pruner


def _optuna_objective(trial):
    current_step_of_complete_nested_cross_validation = 0
    validation_metric_history = []
    fi_pruned = False

    # outer cross-validation
    loo = LeaveOneOut()
    count = 0
    for remain_index, test_index in loo.split(data):  # pylint:disable=unused-variable
        count += 1
        x_remain = data.iloc[remain_index, :]
        y_remain = label.iloc[remain_index]

        # test splits to determine test metric
        # x_test = data.iloc[test_index, :]
        # y_test = label.iloc[test_index]

        # inner cross-validation
        k_fold_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True)
        for train_index, validation_index in k_fold_cv.split(x_remain, y_remain):
            # count steps starting with 1
            current_step_of_complete_nested_cross_validation += 1
            # print('step: ', current_step_of_complete_nested_cross_validation, 'trial:', trial.number)

            x_train = x_remain.iloc[train_index, :]
            x_validation = x_remain.iloc[validation_index, :]
            y_train = y_remain.iloc[train_index]
            y_validation = y_remain.iloc[validation_index]

            train_data = lgb.Dataset(x_train, label=y_train)
            validation_data = lgb.Dataset(x_validation, label=y_validation)

            # parameters for model training to combat overfitting
            parameters = dict(
                min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 2, math.floor(data.shape[0] / 2)),
                lambda_l1=trial.suggest_float("lambda_l1", 0.0, 3),
                min_gain_to_split=trial.suggest_float("min_gain_to_split", 0, 5),
                max_depth=trial.suggest_int("max_depth", 2, 20),
                bagging_fraction=trial.suggest_float("bagging_fraction", 0.1, 1.0),
                bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
                extra_trees=True,
                objective="binary",
                # metric="binary_logloss",
                metric="l1",
                boosting_type="rf",
                verbose=-1,
            )

            # num_leaves must be greater than 2^max_depth
            max_num_leaves = 2 ** parameters["max_depth"] - 1
            if max_num_leaves < 90:
                parameters["num_leaves"] = trial.suggest_int("num_leaves", 2, max_num_leaves)
            else:
                parameters["num_leaves"] = trial.suggest_int("num_leaves", 2, 90)

            eval_result = {}
            model = lgb.train(
                parameters,
                train_data,
                valid_sets=[validation_data],
                callbacks=[lgb.record_evaluation(eval_result)],
            )
            best_score = model.best_score["valid_0"]["l1"]
            # validation_metric_history.append(model.best_score["valid_0"]["binary_logloss"])
            validation_metric_history.append(best_score)

            trial.report(np.mean(validation_metric_history), current_step_of_complete_nested_cross_validation)
        # trial.report(trim_mean(validation_metric_history, proportiontocut=0.2), count)
        # selected_features = model.feature_importance(importance_type="gain")
        # no_model_build_pruner.communicate_feature_values(selected_features)
        # if trial.should_prune():
        #     print("pruned")
        #     # raise TrialPruned
        #     # return np.mean(validation_metric_history)
        #     return trim_mean(validation_metric_history, proportiontocut=0.2)

    # return trim_mean(validation_metric_history, proportiontocut=0.2)
    trial.set_user_attr("raw_evaluation_metric_list", validation_metric_history)
    return np.mean(validation_metric_history)


def main():
    pruner, no_model_build_pruner = combined_cv_pruner_factory(
        inner_cv_folds=inner_folds,
        threshold=0.36,
        comparison_based_pruner=PercentilePruner(
            percentile=25,
            n_startup_trials=1,
            n_warmup_steps=10,
            n_min_trials=2,
        ),
        base_pruner=PercentilePruner(
            percentile=25,
            n_startup_trials=1,
            n_warmup_steps=10,
            n_min_trials=2,
        ),
        extrapolation_method=Method.OPTIMAL_METRIC,
    )
    # pruner, no_model_build_pruner = combined_cv_pruner_factory(
    #     inner_cv_folds=inner_folds,
    #     threshold=0.25,
    #     comparison_based_pruner=SuccessiveHalvingPruner(
    #         min_resource="auto",
    #         reduction_factor=3,
    #         min_early_stopping_rate=2,
    #         bootstrap_count=0,
    #     ),
    #     base_pruner=SuccessiveHalvingPruner(
    #         min_resource="auto",
    #         reduction_factor=3,
    #         min_early_stopping_rate=2,
    #         bootstrap_count=0,
    #     ),
    #     extrapolation_method=Method.OPTIMAL_METRIC,
    # )
    # optuna.study.delete_study("optuna_study_colon_mean_cumulated_reported", "sqlite:///optuna_pruner_vis.db")
    study = optuna.create_study(
        storage="sqlite:///optuna_pruner_vis.db",
        study_name="optuna_study_colon_mean_cumulated_reported",
        direction="minimize",
        # pruner=pruner,
        load_if_exists=True,
    )

    start_time = datetime.datetime.now()
    study.optimize(
        _optuna_objective,
        n_trials=40,  # number of trials to calculate
        n_jobs=8,
    )
    stop_time = datetime.datetime.now()
    print("duration:", stop_time - start_time)

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.show()

    assert len(study.best_trial.intermediate_values) == data.shape[0] * inner_folds


if __name__ == "__main__":
    main()
