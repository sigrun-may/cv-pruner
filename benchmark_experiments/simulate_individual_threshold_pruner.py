# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""TODO: add docstring."""

import math
import sys
from statistics import mean, median

import numpy as np
import optuna
import settings
from optuna.trial import TrialState
from scipy.stats import trim_mean

import cv_pruner


RESTRICTED_NUMBER_OF_CONSIDERED_TRIALS = settings.trials_per_study
# RESTRICTED_NUMBER_OF_CONSIDERED_STUDIES = settings.studies_per_experiment
RESTRICTED_NUMBER_OF_CONSIDERED_STUDIES = None
DB = settings.db_path

STANDARD_PRUNED = "optuna_pruned"


def evaluate_falsely_pruned_trials_threshold(trial, threshold, result, step_pruned):
    """TODO: add docstring."""
    falsely_pruned = False
    raw_evaluation_metric_list = trial.user_attrs["raw_evaluation_metric_list"]
    if mean(raw_evaluation_metric_list) <= threshold:
        result["mean_pruned_below_threshold_list"].append(mean(raw_evaluation_metric_list))
        falsely_pruned = True
    if median(raw_evaluation_metric_list) <= threshold:
        result["median_pruned_below_threshold_list"].append(median(raw_evaluation_metric_list))
        falsely_pruned = True
        # print("##############################################################")
        # print("trial", trial.value)
        # print("study", result["study_best_value"])
        # print("threshold", threshold)
    if median(raw_evaluation_metric_list) <= result["study_best_value"]:
        three_layer_pruner_fails_list = result["three_layer_pruner_fails_list"]
        three_layer_pruner_fails_list.append(median(raw_evaluation_metric_list))
        result["three_layer_pruner_fails_list"] = three_layer_pruner_fails_list
        # result["three_layer_pruner_fails_list"].append(median(raw_evaluation_metric_list))
        print('result["three_layer_pruner_fails_list"]', result["three_layer_pruner_fails_list"])
        print('result["study_best_value"]', result["study_best_value"])
        print("##############################################################")
        falsely_pruned = True
    # Was the globally best trial pruned, if it was below the threshold?
    if trim_mean(raw_evaluation_metric_list, 0.2) <= threshold:
        result["trimmed_mean_pruned_below_threshold_list"].append(trim_mean(raw_evaluation_metric_list, 0.2))
        # print("##############################################################")
        # print("trial", trial.value)
        # print("study", result["study_best_value"])
        # print("threshold", threshold)
        # if trial.value <= result["study_best_value"]:
        #     result["three_layer_pruner_fails_list"].append(trial.value)
        #     print('result["three_layer_pruner_fails_list"]', result["three_layer_pruner_fails_list"])
        #     print("##############################################################")
        falsely_pruned = True
    if falsely_pruned:
        result["step_false_pruning_list"].append(step_pruned)
        # print("false:")
    # pprint(result)


def evaluate_standard_pruner_fails(current_trial, result):
    assert result["study_best_value"] is not None
    # TODO auf min oder max prüfen und anpassen
    if current_trial.value < result["study_best_value"]:
        result["standard_pruner_fails_list"].append(current_trial.value)


def evaluate_additional_pruning(result, step_pruned_fi, step_pruned_threshold, trial, threshold):
    result["steps_three_layer_pruner"] += min(step_pruned_fi, step_pruned_threshold)
    if step_pruned_fi > step_pruned_threshold:
        result["number_of_trials_pruned_threshold"] += 1
        evaluate_falsely_pruned_trials_threshold(trial, threshold, result, step_pruned_threshold)
    else:
        result["number_of_trials_pruned_fi"] += 1


def evaluate_trial(trial, threshold, method, result):
    # standard pruner
    step_pruned_standard = math.inf
    if trial.state == TrialState.PRUNED:
        # steps were counted zero based
        step_pruned_standard = (trial.last_step + 1) * settings.folds_inner_cv
    elif STANDARD_PRUNED in trial.user_attrs:
        # steps were counted zero based
        step_pruned_standard = (trial.user_attrs[STANDARD_PRUNED] + 1) * settings.folds_inner_cv

    # additional pruners
    step_pruned_threshold = simulate_threshold_pruner(trial, threshold, method)
    step_pruned_fi = simulate_fi_pruner(trial)

    # Was trial standard pruned?
    if step_pruned_standard < math.inf:
        result["steps_standard_pruner"] += step_pruned_standard
        result["number_of_trials_pruned_by_asha"] += 1
        evaluate_standard_pruner_fails(trial, result)

        # Was trial earlier pruned by additional pruner?
        if min(step_pruned_fi, step_pruned_threshold) < step_pruned_standard:
            result["number_of_pruned_trials_later_pruned_by_asha"] += 1
            evaluate_additional_pruning(result, step_pruned_fi, step_pruned_threshold, trial, threshold)
        # Trial was earlier pruned by standard pruner
        else:
            result["number_of_trials_pruned_by_asha_only"] += 1
            result["steps_three_layer_pruner"] += result["steps_per_trial"]

    # trial was not pruned by standard pruner
    else:
        result["steps_standard_pruner"] += result["steps_per_trial"]

        # was trial pruned by additional pruners?
        if min(step_pruned_fi, step_pruned_threshold) < math.inf:
            evaluate_additional_pruning(result, step_pruned_fi, step_pruned_threshold, trial, threshold)

        # trial was not pruned
        else:
            result["number_of_unpruned_trials"] += 1
            result["steps_three_layer_pruner"] += result["steps_per_trial"]


def simulate_fi_pruner(trial):
    return (
        trial.user_attrs["would_prune_no_features_selected"]
        if "would_prune_no_features_selected" in trial.user_attrs.keys()
        else math.inf
    )


def simulate_threshold_pruner(trial, threshold, method):
    # simulate distance based pruning

    assert "raw_evaluation_metric_list" in trial.user_attrs.keys()
    raw_evaluation_metric_list = trial.user_attrs["raw_evaluation_metric_list"]

    # simulate model validation
    for current_cross_validation_step in range(len(raw_evaluation_metric_list)):
        # change counting steps to one based
        current_cross_validation_step += 1
        if cv_pruner.should_prune_against_threshold(
            settings.folds_inner_cv,
            raw_evaluation_metric_list[:current_cross_validation_step],
            threshold,
            start_step=0,
            stop_step=sys.maxsize,
            direction_to_optimize_is_minimize=settings.direction_to_optimize_is_minimize,
            method=method,
            optimal_metric_value=settings.optimal_metric,
        ):
            return current_cross_validation_step

    # trial was not pruned
    return math.inf


def analyze_study(study_name, method, threshold):
    result = {
        "steps_standard_pruner": 0,
        "steps_three_layer_pruner": 0,
        "steps_per_trial": 0,
        "study_best_value": None,
        "number_of_trials_pruned_by_asha_only": 0,
        "number_of_pruned_trials_later_pruned_by_asha": 0,
        "number_of_trials_pruned_by_asha": 0,
        "number_of_trials_pruned_fi": 0,
        "number_of_trials_pruned_threshold": 0,
        "number_of_unpruned_trials": 0,
        "mean_pruned_below_threshold_list": [],
        "median_pruned_below_threshold_list": [],
        "trimmed_mean_pruned_below_threshold_list": [],
        "step_false_pruning_list": [],
        "duration_of_completed_trial_list": [],
        "standard_pruner_fails_list": [],
        "three_layer_pruner_fails_list": [],
    }
    # load study
    study = optuna.create_study(
        storage=DB,
        study_name=study_name,
        load_if_exists=True,
        direction=settings.direction_to_optimize,
    )

    if RESTRICTED_NUMBER_OF_CONSIDERED_TRIALS is None:
        trials = study.get_trials()
    else:
        trials = study.get_trials()[:RESTRICTED_NUMBER_OF_CONSIDERED_TRIALS]

    # # initialize
    for trial in trials:
        if trial.state == TrialState.COMPLETE:
            # TODO adapt to min max
            if result["study_best_value"] is None:
                result["study_best_value"] = math.inf
                # corrected 0 based _trial.last_step to 1 based
                # initialize number of steps for one complete trial
                result["steps_per_trial"] = (trial.last_step + 1) * settings.folds_inner_cv

            best_value_of_trial = np.median(trial.user_attrs["raw_evaluation_metric_list"])
            result["study_best_value"] = min(best_value_of_trial, result["study_best_value"])
            # result["study_best_value"] = trial.value
            # result["study_best_value"] = min(trial.value, result["study_best_value"])

    # fig = optuna.visualization.plot_intermediate_values(study)
    # fig.show()

    # collect pruner data
    for trial in trials:

        if trial.state == TrialState.COMPLETE or trial.state == TrialState.PRUNED:
            steps_standard = result["steps_standard_pruner"]
            steps_three_layer = result["steps_three_layer_pruner"]
            evaluate_trial(trial, threshold, method, result)

            assert result["steps_standard_pruner"] > steps_standard
            assert result["steps_three_layer_pruner"] > steps_three_layer

            assert result["steps_standard_pruner"] - steps_standard <= result["steps_per_trial"]
            assert result["steps_three_layer_pruner"] - steps_three_layer <= result["steps_per_trial"]
        else:
            print(trial.state)
            print("#################### broken trial #######################")
            continue
        result["duration_of_completed_trial_list"].append(trial.duration)

    # save plots
    # fig = optuna.visualization.plot_intermediate_values(study)
    # fig.write_image(
    #     "/vol/projects/smay/develop/pruner/simulations/plots/"
    #     + summary.study_name
    #     + "_intermediate_values_plot.png"
    # )
    assert (
        result["number_of_trials_pruned_by_asha_only"] + result["number_of_pruned_trials_later_pruned_by_asha"]
        == result["number_of_trials_pruned_by_asha"]
    )
    assert (
        settings.trials_per_study - result["number_of_unpruned_trials"]
        == result["number_of_trials_pruned_fi"]
        + result["number_of_trials_pruned_threshold"]
        + result["number_of_trials_pruned_by_asha_only"]
    )
    return result


def get_results(method, experiment_name, threshold):
    """TODO: add docstring."""
    ##########################################################
    # Iterate over all studies
    ##########################################################
    list_of_study_results = []
    summaries = optuna.get_all_study_summaries(storage=DB)
    for summary in summaries:
        # print("------------------")
        print("study:", summary.study_name)
        # print("trials:", summary.n_trials)

        if experiment_name in summary.study_name and (
            RESTRICTED_NUMBER_OF_CONSIDERED_STUDIES is None
            or len(list_of_study_results) < RESTRICTED_NUMBER_OF_CONSIDERED_STUDIES
        ):
            list_of_study_results.append(analyze_study(summary.study_name, method, threshold))

    return list_of_study_results


# print(get_results(cv_pruner.Method.MEAN_DEVIATION_TO_MEDIAN, "colon_cv_pruner0", 0.5))
