pydocstyle --count cv_pruner examples tests
black cv_pruner examples tests --check --diff
flake8 cv_pruner examples tests
isort cv_pruner examples tests --check --diff
mdformat --check *.md
mypy cv_pruner examples tests
pylint cv_pruner examples
