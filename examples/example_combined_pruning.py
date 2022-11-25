# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Accelerating embedded feature selection with combined pruning."""

import math

import lightgbm as lgb
import optuna
from optuna import TrialPruned
from optuna.pruners import SuccessiveHalvingPruner
from scipy.stats import trim_mean
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from cv_pruner import Method, cv_pruner
from data_loader.data_loader import load_colon_data

# Parse the data
label, data = load_colon_data()


def _optuna_objective(trial):

    validation_metric_history = []
    current_step_of_complete_nested_cross_validation = 0

    # outer cross-validation
    loo = LeaveOneOut()
    for outer_fold_index, (remain_index, test_index) in enumerate(loo.split(data)):  # pylint:disable=unused-variable
        x_remain = data.iloc[remain_index, :]
        y_remain = label.iloc[remain_index]

        # test splits to determine test metric
        # x_test = data.iloc[test_index, :]
        # y_test = label.iloc[test_index]

        # inner cross-validation
        inner_folds = 10
        k_fold_cv = StratifiedKFold(n_splits=inner_folds)
        for train_index, validation_index in k_fold_cv.split(x_remain, y_remain):
            # count steps starting with 1
            current_step_of_complete_nested_cross_validation += 1

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
            validation_metric_history.append(model.best_score["valid_0"]["binary_logloss"])

            selected_features = model.feature_importance(importance_type="gain")
            if cv_pruner.no_features_selected(selected_features):
                raise TrialPruned()

            if cv_pruner.should_prune_against_threshold(
                    folds_inner_cv=inner_folds,
                    validation_metric_history=validation_metric_history,
                    threshold_for_pruning=0.45,
                    start_step=4,
                    stop_step=int(data.shape[0] / 3),
                    direction_to_optimize_is_minimize=True,
                    method=Method.MAX_DEVIATION_TO_MEDIAN,
                    # optimal_metric=0,  # optimal metric for logloss
            ):
                # Report intermediate results before stopping the trial
                trial.report(trim_mean(validation_metric_history, proportiontocut=0.2), outer_fold_index)

                # Return feedback to the optimizer to enable improvement of the optimization.
                return trim_mean(validation_metric_history, proportiontocut=0.2)

        # Report intermediate results: Report the trimmed mean of all previously fully calculated inner folds
        # to reduce the variance of the validation results.
        trial.report(trim_mean(validation_metric_history, proportiontocut=0.2), outer_fold_index)

        # Built in optuna standard pruner handles pruning based on the given intermediate results.
        if trial.should_prune():
            raise TrialPruned()

    # optimize based on 20% trimmed mean to exclude outliers
    return trim_mean(validation_metric_history, proportiontocut=0.2)


study = optuna.create_study(
    study_name="optuna_study",
    direction="minimize",
    pruner=SuccessiveHalvingPruner(
        min_resource="auto",
        reduction_factor=3,
        min_early_stopping_rate=2,
        bootstrap_count=0,
    ),
)

study.optimize(
    _optuna_objective,
    n_trials=40,  # number of trials to calculate
    n_jobs=4,
)

# import joblib
# joblib.dump("result.pkl.gz", study)
# fig = optuna.visualization.plot_intermediate_values(study)
# fig.show()
