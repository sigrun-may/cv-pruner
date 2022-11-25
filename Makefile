src := cv_pruner
test-src := tests
other-src := examples benchmark_experiments data_loader

# check the code
check:
	pydocstyle --count $(src) $(other-src) $(test-src)
	black $(src) $(other-src) $(test-src) --check --diff
	flake8 $(src) $(other-src) $(test-src)
	isort $(src) $(other-src) $(test-src) --check --diff
	mdformat --check *.md
	mypy --install-types --non-interactive $(src) $(other-src) $(test-src)
	pylint $(src) $(other-src)

# format the code
format:
	black $(src) $(other-src) $(test-src)
	isort $(src) $(other-src) $(test-src)
	mdformat *.md
