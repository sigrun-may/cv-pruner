# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""TODO: add docstring."""

import matplotlib.pyplot as plt
import seaborn as sns


NUMBER_OF_TRIALS = 60


def get_title(experiment_name):
    if "colon" in experiment_name:
        title = "Colon Cancer Data"
    elif "prostate" in experiment_name:
        title = "Prostate Cancer Data"
    elif "leukemia" in experiment_name:
        title = "Leukemia Data"
    else:
        raise Exception("Wrong experiment")
    return title


def plot_errors(complete_df, experiment_name):

    # rename column
    plot_df = complete_df.copy()
    plot_df.rename(
        columns={"difference_threshold": "Margin of error falsely pruned trials against threshold"}, inplace=True
    )

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    sns.set(rc={"figure.figsize": (4, 3)})
    g = sns.FacetGrid(plot_df, col="method")
    g.map_dataframe(sns.histplot, x="Margin of error falsely pruned trials against threshold", binwidth=0.0005)
    # g.set(title='Margin of error for falsely pruned trials compared to the threshold')
    plt.suptitle(get_title(experiment_name) + "\n")
    plt.savefig("plots/" + experiment_name + "_hist.pdf")
    plt.show()


def plot_comparisons_in_percent(vis_df, experiment_name):
    # beide Methoden (unten) funktionieren
    # siehe auch https://seaborn.pydata.org/examples/grouped_barplot.html
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    sns.set(rc={"figure.figsize": (11, 5)})  # width=8, height=4
    ax = plt.gca()
    chart = sns.barplot(x="method", y="value", hue="type of comparison", data=vis_df, ax=ax).set(
        title=get_title(experiment_name)
    )
    ax.set_xticklabels(
        labels=[
            "median only",
            "mean deviation\n from median",
            "max deviation\n from median",
            "optimal performance\n evaluation metric",
            "ASHA pruner",
        ],
        rotation=45,
    )
    # for bars in ax.containers:
    #     ax.bar_label(bars, fmt="%.1f")
    # plt.tight_layout()
    ax.set_ylabel("percent %")
    ax.set_xlabel("\n extrapolation strategy of three-layer pruner vs standard pruning")

    for p in ax.patches:
        ax.annotate(
            "%.2f" % p.get_height() + " %",
            (p.get_x() + p.get_width() / 2.0, p.get_height() / 2),
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

    # ax = sns.catplot(kind="bar", x="method", y="value", hue="type", data=vis_df)

    # plt.savefig(
    #     "/vol/projects/smay/develop/pruner/experiments/results/"
    #     + experiment_name
    #     + ".pdf"
    # )

    # plt.savefig("plots/" + experiment_name + "_percent.png")
    plt.show()


def plot_stacked_pruned_trials_per_part(complete_df, experiment_name):
    sns.set_theme(style="whitegrid")

    # create new dataframe for visualisation
    plot_df2 = complete_df[
        [
            "method",
            "number_of_trials_pruned_by_asha_only",
            "number_of_trials_pruned_fi",
            "number_of_trials_pruned_threshold",
            "number_of_unpruned_trials",
        ]
    ]
    plot_df2 = plot_df2.rename(
        columns={"number_of_trials_pruned_by_asha_only": "number of trials\npruned by ASHA only"}
    )
    plot_df2 = plot_df2.rename(columns={"number_of_trials_pruned_fi": "number of semantically\npruned trials"})
    plot_df2 = plot_df2.rename(
        columns={"number_of_trials_pruned_threshold": "number of trials\npruned against threshold"}
    )
    plot_df2 = plot_df2.rename(columns={"number_of_unpruned_trials": "number of unpruned trials"})
    # print(plot_df2.to_latex(index=False))
    grouped_df = plot_df2.groupby(by=["method"]).sum()

    # # print latex table
    # grouped_tables_for_latex(grouped_df)
    # print(grouped_df.index)
    # grouped_df = grouped_df.reindex(["median only", "mean deviation", "max deviation", "optimal evaluation metric"])
    # print(grouped_df.index)

    ax = grouped_df.plot.bar(
        stacked=True,
        color=["k", "steelblue", "lightsteelblue", "slategrey"],
        title=get_title(experiment_name),
        rot=45,
        fontsize=25,
        figsize=(14, 10),
    )
    ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=25)
    # Let's put the annotations inside the bars themselves by using a
    # negative offset.
    y_offset = -15
    # For each patch (basically each rectangle within the bar), add a label.
    for bar in ax.patches:
        ax.text(
            # Put the next to each bar. get_x returns the start
            bar.get_x() + bar.get_width() + bar.get_width() / 7,
            # Vertically, add the height of the bar to the start of the bar,
            # along with the offset.
            bar.get_height() + bar.get_y() + y_offset,
            # This is actual value we'll show.
            round(bar.get_height()),
            # Center the labels and style them a bit.
            ha="center",
            color="k",
            weight="bold",
            size=12,
        )
    ax.set_xticklabels(
        labels=[
            "median only",
            "mean deviation\nfrom median",
            "max deviation\nfrom median",
            "optimal performance\nevaluation metric",
        ],
        rotation=45,
    )
    ax.set_ylabel("number of trials", fontsize=25)
    ax.set_xlabel("\n extrapolation method the three-layer pruner", fontsize=25)
    # ax.set_xlabel("\n combined parts of the three-layer pruner", fontsize=25)
    # displaying the title
    plt.title(get_title(experiment_name), fontsize=30)
    ax.plot()
    plt.savefig("plots/" + experiment_name + "_combined_pruner.pdf")
    plt.show()


def tables_for_latex(df):
    print(df.to_latex(index=False))


def grouped_tables_for_latex(grouped_df):
    grouped_df.insert(loc=0, value=grouped_df.index, column="method")
    print(grouped_df.to_latex(index=False))
