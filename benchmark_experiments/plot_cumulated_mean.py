"""TODO: add docstring."""

from statistics import mean, median

import matplotlib.patches as mpatches
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from scipy.stats import trim_mean

import settings

DB = "sqlite:///optuna_paper_db_node4.db"
k_folds = settings.folds_inner_cv

summaries = optuna.study.get_all_study_summaries(storage=DB)
for summary in summaries:
    print(summary.study_name)

    study = optuna.create_study(
        storage=DB,
        # study_name=settings.study_name,
        study_name=summary.study_name,
        load_if_exists=True,
        # direction='is_minimize',
        direction=settings.direction_to_optimize,
    )

    trials = study.get_trials()

    for trial in trials:
        if np.sum(np.asarray(trial.user_attrs["number_of_features_in_model_list"])) > 0:
            print(trial.number)
            mean_raw_value = []
            trim_mean_raw_value = []
            for j in range(1, len(trial.user_attrs["raw_evaluation_metric_list"]) + 1):
                mean_raw_value.append(mean(trial.user_attrs["raw_evaluation_metric_list"][:j]))
                trim_mean_raw_value.append(trim_mean(trial.user_attrs["raw_evaluation_metric_list"][:j], 0.2))

            evaluation_metric_df = pd.DataFrame()
            evaluation_metric_df["logloss"] = trial.user_attrs["raw_evaluation_metric_list"]
            evaluation_metric_df["mean logloss"] = mean_raw_value
            evaluation_metric_df["trim_mean logloss"] = trim_mean_raw_value
            evaluation_metric_df["step of nested cross-validation"] = np.array(
                range(len(trial.user_attrs["raw_evaluation_metric_list"]))) + 1

            raw_evaluation_metric_list = trial.user_attrs["raw_evaluation_metric_list"]
            reduced_variance_list = []
            reduced_variance_list2 = []
            median_list = []
            mean_list = []
            cumulated_mean_list = []  # type: ignore  # FIXME: this is never used

            for i in range(1, len(raw_evaluation_metric_list) + 1):
                if i % k_folds == 0:
                    median_list.append(median(raw_evaluation_metric_list[:i]))
                    mean_list.append(mean(raw_evaluation_metric_list[:i]))
                    reduced_variance_list.append(trim_mean(raw_evaluation_metric_list[:i], 0.2))
                    reduced_variance_list2.append(mean(raw_evaluation_metric_list[i - k_folds: i]))
                    # reduced_variance_list2.append(
                    #     trim_mean(raw_evaluation_metric_list[i-k_folds: i], 0.2)
                    # )

            # for i in range(1, len(raw_evaluation_metric_list)+1):
            #     print('i', i)
            #     if i % k_folds == 0 and i > 0:
            #         # print(len(raw_evaluation_metric_list[:i]))
            #         median_list.append(median(raw_evaluation_metric_list[:i]))
            #         mean_list.append(mean(raw_evaluation_metric_list[:i]))
            #         reduced_variance_list.append(
            #             trim_mean(raw_evaluation_metric_list[:i], 0.2)
            #         )

            # assert len(reduced_variance_list) == len(list(((np.asarray(range(len(reduced_variance_list)))) + 1) * 8))  # noqa: E501
            # print(len(reduced_variance_list))
            # print(len(list(((np.asarray(range(len(reduced_variance_list)))) + 1) * 8)))
            # print('list', list(((np.asarray(range(len(reduced_variance_list)))) + 1) * 8))

            cumulated_evaluation_metric_df = pd.DataFrame()
            cumulated_evaluation_metric_df["cummulated logloss"] = reduced_variance_list
            cumulated_evaluation_metric_df["cummulated logloss2"] = reduced_variance_list2
            cumulated_evaluation_metric_df["cummulated median logloss"] = median_list
            cumulated_evaluation_metric_df["cummulated mean logloss"] = mean_list
            cumulated_evaluation_metric_df["outer folds of nested cross-validation"] = (
                                                                                               (np.asarray(range(
                                                                                                   len(reduced_variance_list)))) + 1
                                                                                       ) * k_folds
            assert len(reduced_variance_list) == len(
                cumulated_evaluation_metric_df["outer folds of nested cross-validation"]
            )

            sns.set(rc={'figure.figsize': (20, 16)})
            sns.scatterplot(
                y=evaluation_metric_df["logloss"],
                x=evaluation_metric_df["step of nested cross-validation"],
                color="orange",
            ).set(title="Reduced variance of evaluation metric")

            sns.lineplot(
                y=evaluation_metric_df["mean logloss"],
                x=evaluation_metric_df["step of nested cross-validation"],
                color="red",
            )
            # sns.lineplot(
            #     y=evaluation_metric_df["mean logloss"],
            #     x=evaluation_metric_df["step of nested cross-validation"],
            #     color="red",
            # )
            sns.lineplot(
                y=evaluation_metric_df["trim_mean logloss"],
                x=evaluation_metric_df["step of nested cross-validation"],
                color="brown",
            )
            sns.lineplot(
                y=cumulated_evaluation_metric_df["cummulated median logloss"],
                x=cumulated_evaluation_metric_df[
                    "outer folds of nested cross-validation"
                ],
                color="green",
            )
            sns.lineplot(
                y=cumulated_evaluation_metric_df["cummulated mean logloss"],
                x=cumulated_evaluation_metric_df[
                    "outer folds of nested cross-validation"
                ],
                color="black",
            )
            sns.lineplot(
                y=cumulated_evaluation_metric_df["cummulated logloss"],
                x=cumulated_evaluation_metric_df[
                    "outer folds of nested cross-validation"
                ],
                color="blue",
            )
            # sns.lineplot(
            #     y=cumulated_evaluation_metric_df["cummulated logloss2"],
            #     x=cumulated_evaluation_metric_df[
            #         "outer folds of nested cross-validation"
            #     ],
            #     color="purple",
            # )
            orange_patch = mpatches.Patch(color="orange", label="individual objective value")
            red_patch = mpatches.Patch(color="red", label="cumulated mean objective value")
            blue_patch = mpatches.Patch(color="blue",
                                        label="cumulated 20% trimmed mean objective value\nfor each iteration of outer cross-validation only")
            green_patch = mpatches.Patch(color="green",
                                         label="cumulated median objective value\nfor each iteration of outer cross-validation only")
            black_patch = mpatches.Patch(color="black",
                                         label="cumulated mean objective\nfor each iteration of outer cross-validation only")
            # pyplot.legend(handles=[red_patch])  # cumulated mean only
            pyplot.legend(handles=[orange_patch, red_patch, blue_patch, green_patch, black_patch])
            pyplot.show()
