# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

##########################
# settings for optuna
##########################

"""TODO: add docstring."""

n_trials = 40

# set direction to optimize
direction_to_optimize_is_minimize = True
if direction_to_optimize_is_minimize:
    direction_to_optimize = "minimize"
else:
    direction_to_optimize = "maximize"

n_jobs_optuna = 6  # number of parallel processes for optuna
if n_jobs_optuna > 1:
    show_progress = False
else:
    show_progress = True

##########################
# settings for lightgbm
##########################
boosting_type = "rf"
objective = "binary"
metric = "binary_logloss"
device = "cpu"
num_cores = 1  # number of parallel processes for lightgbm

##########################
# settings for pruning
##########################
optimal_metric = 0
simulate_standard_pruner = True

##########################
# settings for nested cross validation
##########################
folds_inner_cv = 10
reduce_variance = True

##########################
# define logging level
##########################
logging_level = "DEBUG"
