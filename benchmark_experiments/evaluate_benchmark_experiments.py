import optuna


DB = "sqlite:///optuna_pruner_vis.db"


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
        print("trials:", summary.n_trials)

        # load study
        study = optuna.create_study(
            storage=DB,
            study_name=summary.study_name,
            load_if_exists=True,
            direction="minimize",
        )

        # if RESTRICTED_NUMBER_OF_CONSIDERED_TRIALS is None:
        #     trials = study.get_trials()
        # else:
        #     trials = study.get_trials()[:RESTRICTED_NUMBER_OF_CONSIDERED_TRIALS]

        # # initialize
        trials = study.get_trials()
        count = 0
        for trial in trials:
            if trial.user_attrs:
                count += 1
                print("trial", trial.number)
                for attr in trial.user_attrs:
                    print(attr)
                    print(trial.user_attrs[attr])
                print("##########################")
        print("pruned trials: ", count)

    #     if experiment_name in summary.study_name and (
    #         RESTRICTED_NUMBER_OF_CONSIDERED_STUDIES is None
    #         or len(list_of_study_results) < RESTRICTED_NUMBER_OF_CONSIDERED_STUDIES
    #     ):
    #         list_of_study_results.append(analyze_study(summary.study_name, method, threshold))
    #
    # return list_of_study_results


get_results("n", "n", 0)
