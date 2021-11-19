pydocstyle --count cv_pruner examples tests
black cv_pruner examples tests --check --diff
flake8 cv_pruner examples tests
isort cv_pruner examples tests --check --diff
mdformat --check README.md
mypy --install-types --non-interactive cv_pruner examples tests
pylint cv_pruner examples/data_loader.py examples/example_combined_pruning.py
