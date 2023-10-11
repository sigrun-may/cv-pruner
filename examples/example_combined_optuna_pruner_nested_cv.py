# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
import datetime
import math

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna import TrialPruned
from optuna.trial import TrialState
from optuna.pruners import PercentilePruner, SuccessiveHalvingPruner
from scipy.stats import trim_mean
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

import cv_pruner.optuna_pruner
from hpo_tools import optuna_study_pruner
from cv_pruner import Method
from cv_pruner.optuna_pruner import (
    BenchmarkPruneFunctionWrapper,
    MultiPrunerDelegate,
    NoModelBuildPruner,
    RepeatedTrainingPrunerWrapper,
    RepeatedTrainingThresholdPruner,
)
from data_loader.data_loader import load_colon_data, load_prostate_data


"""Accelerating embedded feature selection with combined pruning."""


def combined_cv_pruner_factory(inner_cv_folds, threshold, extrapolation_method, comparison_based_pruner):
    no_model_build_pruner = NoModelBuildPruner()
    # no_model_build_pruner_benchmark = BenchmarkPruneFunctionWrapper(no_model_build_pruner)

    threshold_pruner = RepeatedTrainingThresholdPruner(
        threshold=threshold,
        extrapolation_interval=inner_cv_folds,
        active_until_step=inner_cv_folds,
        extrapolation_method=extrapolation_method,
    )
    # threshold_pruner = BenchmarkPruneFunctionWrapper(threshold_pruner)

    # comparison_based_pruner = RepeatedTrainingPrunerWrapper(
    #     pruner=comparison_based_pruner, inner_cv_folds=inner_cv_folds
    # )  # no not switch order with BenchmarkPruneFunctionWrapper!
    # comparison_based_pruner = BenchmarkPruneFunctionWrapper(
    #     comparison_based_pruner
    # )  # no not switch order with RepeatedTrainingPrunerWrapper!

    compound_pruner = MultiPrunerDelegate(pruner_list=[no_model_build_pruner, threshold_pruner, comparison_based_pruner], prune_eager=True)

    return compound_pruner


def _optuna_objective(trial):
    if (trial.study.user_attrs["valid"] and (optuna_study_pruner.study_patience_pruner(
        trial, epsilon=0.001, warm_up_steps=15, patience=5
    ) or optuna_study_pruner.study_no_improvement_pruner(
        trial,
        epsilon=0.01,
        warm_up_steps=15,
        number_of_similar_best_values=5,
        threshold=0.3,
    ))):
        print("study stopped")
        trial.study.stop()
        raise TrialPruned()

    validation_metric_history = []

    # inner cross-validation
    k_fold_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True)
    for train_index, validation_index in k_fold_cv.split(x_remain, y_remain):

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
            metric="binary_logloss",
            # metric="l1",
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
        # best_score = model.best_score["valid_0"]["l1"]
        best_score = model.best_score["valid_0"]["binary_logloss"]
        # validation_metric_history.append(model.best_score["valid_0"]["binary_logloss"])
        validation_metric_history.append(best_score)

        trial.report(np.median(validation_metric_history), len(validation_metric_history))
        selected_features = model.feature_importance(importance_type="gain")
        assert len(selected_features) == x_remain.shape[1]
        trial.set_user_attr("selected_features", selected_features)
        # if np.sum(selected_features != 0) == 0 or trial.should_prune():
        if trial.should_prune():
            # raise TrialPruned
            print(f"pruned trial {trial.number} at step {len(validation_metric_history)}")
            trial_state = TrialState.PRUNED
            trial.TrialState = trial_state
            return np.median(validation_metric_history)

        # if np.sum(selected_features != 0) == 0 and not fi_pruned:
        #     print(f"pruned trial {trial.number} at step {len(validation_metric_history)} no features selected")
        #     fi_pruned = True
        #     return np.median(validation_metric_history)
    print("trial complete", trial.number)
    study.set_user_attr("valid", True)
    assert len(validation_metric_history) == inner_folds, f"inner_folds {inner_folds}"
    return np.median(validation_metric_history)


# Parse the data
# df = pd.read_csv("/home/sigrun/PycharmProjects/reverse_feature_selection/data/small_50.csv")
# label = df["label"]
# data = df.iloc[:, 1:]
data, label = load_prostate_data()
# data = data1.iloc[:, :40]
inner_folds: int = 10

pruner = combined_cv_pruner_factory(
    inner_cv_folds=inner_folds,
    threshold=0.4,
    comparison_based_pruner=SuccessiveHalvingPruner(
        min_resource="auto",
        reduction_factor=3,
        min_early_stopping_rate=2,
        bootstrap_count=0,
    ),
    extrapolation_method=Method.OPTIMAL_METRIC,
)

start_time = datetime.datetime.now()

# outer cross-validation
loo = LeaveOneOut()
for remain_index, test_index in loo.split(data):  # pylint:disable=unused-variable
    x_remain = data.iloc[remain_index, :]
    y_remain = label.iloc[remain_index]

    # test splits to determine test metric
    # x_test = data.iloc[test_index, :]
    # y_test = label.iloc[test_index]

    study = optuna.create_study(
        study_name="optuna_study",
        direction="minimize",
        pruner=pruner,
    )
    study.set_user_attr("valid", False)
    study.optimize(
        _optuna_objective,
        n_trials=40,  # number of trials to calculate
        n_jobs=1,
    )
    print(study.best_trial)
    assert len(study.best_trial.intermediate_values) == inner_folds, str(len(study.best_trial.intermediate_values))

stop_time = datetime.datetime.now()
print("duration:", stop_time - start_time)
