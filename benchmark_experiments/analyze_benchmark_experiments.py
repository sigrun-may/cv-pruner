import math
from datetime import datetime, timedelta

import numpy as np
import optuna
from optuna.trial import TrialState

DB = "sqlite:///hpo_flow_pruner.db"


def analyze_study(study_name):
    print(study_name)
    # load study
    study = optuna.create_study(
        storage=DB,
        study_name=study_name,
        load_if_exists=True,
        direction="minimize",
    )

    trials = study.get_trials()
    print('number of trials:', len(trials))
    trial_durations = []

    false_threshold_pruned_trials_margins_of_errors = []

    duration_combined_pruner = timedelta(microseconds=0)
    duration_baseline_pruner = timedelta(microseconds=0)

    number_of_baseline_pruned_trials = 0
    number_of_comparison_based_pruned_trials = 0
    number_of_no_model_pruned_trials = 0
    number_of_threshold_pruned_trials = 0
    number_of_combined_unpruned_trials = 0
    number_of_combined_pruned_trials = 0

    # initialize median_of_best_finished_trial with median of first trial
    # median_of_best_finished_trial = np.median(list(trials[0].intermediate_values.values()))
    median_of_best_finished_trial = math.inf

    for trial in trials:
        if trial.state != TrialState.COMPLETE:
            continue

        trial_durations.append(trial.duration)

        baseline_pruned = False
        combined_pruned = False

        median_of_trial = np.median(list(trial.intermediate_values.values()))
        min_time = datetime.now()
        min_step = math.inf
        fastest_pruner = ""

        # was this trial pruned by any pruner?
        any_pruner_pruned = False
        for user_attr_name in trial.user_attrs.keys():
            if user_attr_name.endswith("_pruned_at"):
                any_pruner_pruned = True
                break

        if any_pruner_pruned:
            print(trial.number, np.median(list(trial.intermediate_values.values())), trial.user_attrs)
            for pruner, result in trial.user_attrs.items():
                if pruner.endswith("_pruned_at"):
                    # if "comparison" in pruner:
                    if "hpo_flow_pruner" in pruner:
                        duration = datetime.fromisoformat(result['timestamp']) - trial.datetime_start
                        duration_baseline_pruner += duration
                        baseline_pruned = True
                        number_of_baseline_pruned_trials += 1
                        best_trial_numbers = [best_trial.number for best_trial in study.best_trials]
                        print("step", result['step'])

                        if trial.number == study.best_trial.number:
                            print(f"-------------- best trial pruned baseline {np.median(list(study.best_trial.intermediate_values.values()))} --------------")
                            print("step", result['step'])
                            print('median', np.median(list(study.best_trial.intermediate_values.values())[:result['step']]))
                            print("---------")

                    # else:
                    if datetime.fromisoformat(result['timestamp']) < min_time:
                        fastest_pruner = pruner
                        min_step = result['step']
                        min_time = datetime.fromisoformat(result['timestamp'])
                        combined_pruned = True

            # check if trial was correctly pruned against the threshold
            if "Threshold" in fastest_pruner and (median_of_trial <= study.user_attrs['threshold']):
                # calculate margin of error
                difference = study.user_attrs['threshold'] - median_of_trial
                false_threshold_pruned_trials_margins_of_errors.append(difference)
                print('false pruned')
                print('median_of_trial', median_of_trial)
                print('difference', difference)

        if not baseline_pruned:
            duration_baseline_pruner += trial.duration
        if combined_pruned:
            if trial.number == study.best_trial.number:
                print(f"!!!!!!!!!!!!!!! best trial pruned {fastest_pruner} { np.median(list(study.best_trial.intermediate_values.values()))} !!!!!!!!!!!!!!!!!!!!!")
                print(min_step)
                print('median', np.median(list(study.best_trial.intermediate_values.values())[:min_step]))

            duration = min_time - trial.datetime_start
            duration_combined_pruner += duration
            if "NoModelBuildPruner" in fastest_pruner:
                number_of_no_model_pruned_trials += 1
            if "RepeatedTrainingThresholdPruner" in fastest_pruner:
                number_of_threshold_pruned_trials += 1
            if "comparison" in fastest_pruner:
                number_of_comparison_based_pruned_trials += 1

            number_of_combined_pruned_trials += 1
        else:  # nothing pruned
            duration_combined_pruner += trial.duration
            number_of_combined_unpruned_trials += 1

            # calculate the median of the best unpruned trial
            median_of_best_finished_trial = min(median_of_trial, median_of_best_finished_trial)

    assert len(
        trial_durations) == number_of_combined_pruned_trials + number_of_combined_unpruned_trials, f"{len(trial_durations)}, {number_of_combined_pruned_trials}, {number_of_combined_unpruned_trials}"
    print(study_name, "---------------")
    print("number_of_combined_pruned_trials", number_of_combined_pruned_trials)
    print("duration_combined_pruner", duration_combined_pruner)
    print("duration_baseline_pruner", duration_baseline_pruner)
    print("number_of_baseline_pruned_trials", number_of_baseline_pruned_trials)
    print("number_of_no_model_pruned_trials", number_of_no_model_pruned_trials)
    print("number_of_threshold_pruned_trials", number_of_threshold_pruned_trials)
    print("number_of_comparison_based_pruned_trials", number_of_comparison_based_pruned_trials)
    print("number_of_combined_unpruned_trials", number_of_combined_unpruned_trials)
    print("number_of_false_pruned_trials", len(false_threshold_pruned_trials_margins_of_errors))
    print('median_of_best_finished_trial', median_of_best_finished_trial)
    return trial_durations


# results = [
#     ("colon_cv_pruner0", np.sum(np.asarray(analyze_study("colon_cv_pruner0")))),
#     ("prostate_cv_pruner0", np.sum(np.asarray(analyze_study("prostate_cv_pruner0")))),
#     ("leukemia_cv_pruner0", np.sum(np.asarray(analyze_study("leukemia_cv_pruner0")))),
# ]

# list_of_study_results = []
# summaries = optuna.get_all_study_summaries(storage=DB)
# for summary in summaries:
#     # print("------------------")
#     print("study:", summary.study_name)
#     print(summary)

# for study, result in results:
#     print(study)
#     pprint(str(result))
#     # print("min", min(result))
#     # print("max", max(result))
#     # print("mean", np.mean(result))
#     print("-----------------")


summaries = optuna.get_all_study_summaries(storage=DB)
for summary in summaries:
    print("study_duration", np.sum(np.asarray(analyze_study(summary.study_name))))
    print("------------------")
