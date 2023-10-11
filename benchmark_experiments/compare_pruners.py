# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""TODO: add docstring."""

import joblib
import numpy as np
import pandas as pd
import plotter
import settings
import simulate_individual_threshold_pruner as pruner_simulation

import cv_pruner


NUMBER_OF_TRIALS = settings.trials_per_study


def percent(small, big):
    return (np.asarray(small) / np.asarray(big)) * 100


def fill_df(name, result, df, pruner_threshold):
    if "steps_per_trial" in name:
        df["steps_per_study"] = np.full(
            len(results), pruner_simulation.RESTRICTED_NUMBER_OF_CONSIDERED_TRIALS * results[0]["steps_per_trial"]
        )
    elif "trimmed_mean_pruned_below_threshold_list" in name:
        tm_below_th_value = []
        for r in results:
            if len(r["trimmed_mean_pruned_below_threshold_list"]) > 0:
                tm_below_th_value.append(min(r["trimmed_mean_pruned_below_threshold_list"]))
            else:
                tm_below_th_value.append(pruner_threshold)
        df["min_trimmed_mean_pruned_below_threshold"] = tm_below_th_value
        df[name] = [len(r[name]) for r in result]
    elif "list" in name:
        df[name] = [len(r[name]) for r in result]
    else:
        df[name] = [r[name] for r in result]


list_of_experiments = [
    ["colon_cv_pruner", 0.2],
    ["prostate_cv_pruner", 0.2],
    ["leukemia_cv_pruner", 0.2],
]

experiment_count = 0
for experiment_name, threshold in list_of_experiments:
    experiment_count += 1
    first = True
    result_df_list = []

    for method in cv_pruner.Method:
        print(method.name)
        print(experiment_name)
        results = pruner_simulation.get_results(method, experiment_name, threshold)
        joblib.dump(
            results,
            "results/result_" + experiment_name + "_" + "12jobs" + "_" + method.name + ".pkl.gz",
            compress=("gzip", 3),
        )
        # results = joblib.load("results/result_" + experiment_name + "_" + '12jobs' + "_" + method.name + ".pkl.gz")

        # results = joblib.load("results/result_" + experiment_name + "_" + method.name + ".pkl.gz")

        result_df = pd.DataFrame()
        result_df["method"] = np.full(30, method.name.lower())
        for key in results[0].keys():
            fill_df(key, results, result_df, threshold)
            if first and "duration_of_completed_trial_list" in key:
                # trial duration
                print("---")
                durations = results[0]["duration_of_completed_trial_list"]
                print("duration_of_completed_trials", np.sum(durations))
                first = False

        result_df["difference_best_value"] = (
            result_df["min_trimmed_mean_pruned_below_threshold"] - result_df["study_best_value"]
        )
        result_df["difference_threshold"] = np.full(
            len(result_df["min_trimmed_mean_pruned_below_threshold"]), threshold
        ) - np.asarray(result_df["min_trimmed_mean_pruned_below_threshold"])

        result_df_list.append(result_df)

    # concat all methods and all studies to one experiment
    complete_df = pd.concat(result_df_list, ignore_index=True)

    complete_df.replace(
        {
            cv_pruner.Method.MEDIAN.name: "median only",
            cv_pruner.Method.MEAN_DEVIATION_TO_MEDIAN.name: "mean deviation",
            cv_pruner.Method.MAX_DEVIATION_TO_MEDIAN.name: "max deviation",
            cv_pruner.Method.OPTIMAL_METRIC.name: "optimal metric",
        },
        inplace=True,
    )

    # add percentages
    complete_df["number_of_trials_per_study"] = np.full(
        len(complete_df["trimmed_mean_pruned_below_threshold_list"]), NUMBER_OF_TRIALS
    )
    complete_df["trimmed_mean_below_threshold_percent"] = 100 - percent(
        complete_df["trimmed_mean_pruned_below_threshold_list"], complete_df["number_of_trials_per_study"]
    )
    complete_df["fails_standard_pruner_percent"] = 100 - percent(
        complete_df["standard_pruner_fails_list"], complete_df["number_of_trials_per_study"]
    )
    complete_df["global_fails_threshold_pruner_percent"] = 100 - percent(
        complete_df["three_layer_pruner_fails_list"], complete_df["number_of_trials_per_study"]
    )
    assert complete_df["steps_per_study"][0] == complete_df["number_of_trials_per_study"][0] * 300
    complete_df["steps_three_layer_percent"] = 100 - percent(
        complete_df["steps_three_layer_pruner"], complete_df["steps_per_study"]
    )
    complete_df["steps_standard_pruner_percent"] = 100 - percent(
        complete_df["steps_standard_pruner"], complete_df["steps_per_study"]
    )
    assert complete_df["steps_per_study"][0] == complete_df["number_of_trials_per_study"][0] * 300
    complete_df["steps_three_layer_percent"] = 100 - percent(
        complete_df["steps_three_layer_pruner"], complete_df["steps_per_study"]
    )
    complete_df["steps_standard_pruner_percent"] = 100 - percent(
        complete_df["steps_standard_pruner"], complete_df["steps_per_study"]
    )

    # create DataFrame for visualization
    standard_pruner_df1 = pd.DataFrame(
        {
            "method": np.full(30, "ASHA"),
            "value": complete_df["steps_standard_pruner_percent"].iloc[
                (30 * (experiment_count - 1)) : (30 * experiment_count)
            ],
            "type of comparison": np.full(30, "saved iterations"),
        }
    )
    assert len(standard_pruner_df1["value"]) == 30
    standard_pruner_df2 = pd.DataFrame(
        {
            "method": np.full(30, "ASHA"),
            "value": complete_df["fails_standard_pruner_percent"].iloc[
                (30 * (experiment_count - 1)) : (30 * experiment_count)
            ],
            "type of comparison": np.full(30, "correct pruned trials\nagainst threshold"),
        }
    )
    assert len(standard_pruner_df2["value"]) == 30
    standard_pruner_df3 = pd.DataFrame(
        {
            "method": np.full(30, "ASHA"),
            "value": complete_df["fails_standard_pruner_percent"].iloc[
                (30 * (experiment_count - 1)) : (30 * experiment_count)
            ],
            "type of comparison": np.full(30, "completed global best trials\nof full optimization"),
        }
    )
    assert len(standard_pruner_df3["value"]) == 30

    df1 = complete_df[["method", "steps_three_layer_percent"]].copy()
    df1.rename(columns={"steps_three_layer_percent": "value"}, inplace=True)
    # df1.loc[:, "steps_three_layer_percent"] = "value"
    df1["type of comparison"] = "saved iterations"

    df2 = complete_df[["method", "trimmed_mean_below_threshold_percent"]].copy()
    df2.rename(columns={"trimmed_mean_below_threshold_percent": "value"}, inplace=True)
    df2["type of comparison"] = "correct pruned trials\nagainst threshold"

    df3 = complete_df[["method", "global_fails_threshold_pruner_percent"]].copy()
    df3.rename(columns={"global_fails_threshold_pruner_percent": "value"}, inplace=True)
    df3["type of comparison"] = "completed global best trials\nof full optimization"

    vis_df = pd.concat(
        [df1, df2, df3, standard_pruner_df1, standard_pruner_df2, standard_pruner_df3], ignore_index=True
    )
    vis_df.fillna(value=0, inplace=True)

    plotter.plot_errors(complete_df, experiment_name)
    plotter.plot_comparisons_in_percent(vis_df, experiment_name)
    plotter.plot_stacked_pruned_trials_per_part(complete_df, experiment_name)
