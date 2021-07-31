"""TODO: add docstring."""

import math
import warnings

import lightgbm as lgb
import optuna
import pandas as pd
from optuna import TrialPruned
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from scipy.stats import trim_mean
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from cv_pruner import cv_pruner


warnings.filterwarnings("ignore")

# Parse the data
# data_df = pd.read_csv("../data/colon.csv")
data_df = pd.read_csv("../data/huge_data.csv")
label = data_df["label"]
data = data_df.iloc[:, 1:100]

# Calculate Baseline

# Always predicts the probability of the class that maximizes the class prior (like “most_frequent”).
dummy_clf = DummyClassifier(strategy="prior")
dummy_clf.fit(data, label)
prediction = dummy_clf.predict_proba(data)
baseline = log_loss(label, prediction)  # calculate desired optimization metric
print("baseline", baseline)

# Settings
folds_inner_cross_validation_loop = 8
threshold_for_pruning = 0.26
minimize_as_direction_to_optimize = True
accepted_proportion_of_models_below_chance = 0.1
steps_of_ncv = (
    data.shape[0] * folds_inner_cross_validation_loop
)  # outer loop = number of samples (leave one out cross-validation)
accepted_number_of_models_below_chance = steps_of_ncv * accepted_proportion_of_models_below_chance


def embedded_feature_selection(params, trial):
    """TODO: add docstring."""
    # initialize variables
    evaluation_metric_list = []
    current_step_complete_ncv = 0

    loo = LeaveOneOut()
    for outer_sample_index, (remain_index, test_index) in enumerate(loo.split(data)):
        x_remain = data.iloc[remain_index, :]
        y_remain = label.iloc[remain_index]

        # test set is not considered for the simulation of pruning
        # x_test = data.iloc[test_index, :]
        # y_test = label.iloc[test_index]

        k_fold_cv = StratifiedKFold(folds_inner_cross_validation_loop)
        for inner_sample_index, (train_index, validation_index) in enumerate(k_fold_cv.split(x_remain, y_remain)):
            # count steps starting with 1
            current_step_complete_ncv += 1

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
            evaluation_metric_list.append(model.best_score["valid_0"]["binary_logloss"])

            selected_features = model.feature_importance(importance_type="gain")
            if cv_pruner.check_no_features_selected(selected_features):
                raise TrialPruned()

            if cv_pruner.check_against_threshold(
                current_step_complete_ncv,
                int,
                folds_inner_cross_validation_loop,
                evaluation_metric_list,
                threshold_for_pruning,
                minimize_as_direction_to_optimize,
                optimal_metric=0,
                method="optimal",
            ):
                # Return feedback to the optimizer to enable improvement of the optimization.
                return trim_mean(evaluation_metric_list, proportiontocut=0.2)

        # Report intermediate results: Report the trimmed mean of all previously fully calculated inner folds
        # to reduce the variance of the validation results.
        trial.report(trim_mean(evaluation_metric_list, proportiontocut=0.2), outer_sample_index)

        # Built in optuna standard pruner handles pruning based on the given intermediate results.
        if trial.should_prune():
            print("optuna pruned at step: ", current_step_complete_ncv)
            raise TrialPruned()

    # optimize based on 20% trimmed mean to exclude outliers
    return trim_mean(evaluation_metric_list, proportiontocut=0.2)


def optimize():
    """TODO: add docstring."""

    def optuna_objective(trial):
        sample_size = data.shape[0]

        # parameters for model training
        parameters = dict(
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 2, math.floor(sample_size / 2)),
            lambda_l1=trial.suggest_uniform("lambda_l1", 0.0, 3),
            min_gain_to_split=trial.suggest_uniform("min_gain_to_split", 0, 5),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            bagging_fraction=trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
            # extra_trees=trial.suggest_categorical("extra_trees", [True, False]),
            extra_trees=True,
            # num_boost_round=trial.suggest_int("num_boost_round", 75, 150)
            num_boost_round=100,
            feature_pre_filter=False,
            objective="binary",
            metric="binary_logloss",
            boosting_type="rf",
            verbose=-1,
            n_jobs=1,
        )

        # num_leaves must be greater than 2^max_depth
        max_num_leaves = 2 ** parameters["max_depth"] - 1
        if max_num_leaves < 90:
            parameters["num_leaves"] = trial.suggest_int("num_leaves", 2, max_num_leaves)
        else:
            parameters["num_leaves"] = trial.suggest_int("num_leaves", 2, 90)

        return embedded_feature_selection(parameters, trial)

    # Make the sampler behave in a deterministic way by setting a seed.
    # sampler = TPESampler(seed = 10, multivariate = True)

    sampler = TPESampler(multivariate=True)

    optuna.study.delete_study("deleteMeLater", storage="sqlite:///optuna_database.db")
    study = optuna.create_study(
        storage="sqlite:///optuna_database.db",
        study_name="deleteMeLater",
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=SuccessiveHalvingPruner(
            min_resource="auto",
            reduction_factor=3,
            min_early_stopping_rate=2,
            bootstrap_count=0,
        ),
    )

    study.optimize(
        optuna_objective,
        n_trials=40,  # number of trials to calculate
        n_jobs=1,  # number of parallel trials
    )

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.show()


optimize()
