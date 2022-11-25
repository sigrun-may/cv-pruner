# Copyright (c) 2021 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2021 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Build script for setuptools."""

import os

import setuptools

project_name = "cv_pruner"
source_code = "https://github.com/sigrun-may/cv-pruner"
keywords = "ml ai machine-learning hyperparameter-optimization high-dimensional data"
install_requires = ["numpy"]
extras_require = {
    "checking": [
        "black",
        "flake8",
        "isort",
        "mdformat",
        "pydocstyle",
        "mypy",
        "pylint",
        "pylintfileheader",
    ],
    "optional": ["pandas", "optuna", "scipy", "lightgbm", "tqdm", "beautifulsoup4"],
    "testing": ["pytest"],
    "doc": ["sphinx", "sphinx_rtd_theme", "myst_parser", "sphinx_copybutton"],
}

# add "all"
all_extra_packages = list(
    {package_name for value in extras_require.values() for package_name in value}
)
extras_require["all"] = all_extra_packages


def get_version():
    """Read version from ``__init__.py``."""
    version_filepath = os.path.join(os.path.dirname(__file__), project_name, "__init__.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=project_name,
    version=get_version(),
    maintainer="Sigrun May",
    author="Sigrun May",
    author_email="s.may@ostfalia.de",
    description="Three-layer Pruning for Nested Cross-Validation to Accelerate Automated Hyperparameter Optimization"
    " for Embedded Feature Selection in High-Dimensional Data With Very Small Sample Sizes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=source_code,
    project_urls={
        "Bug Tracker": source_code + "/issues",
        # "Documentation": "",  # TODO: add this
        "Source Code": source_code,
        "Contributing": source_code + "/blob/main/CONTRIBUTING.md",  # TODO: add this file later
        "Code of Conduct": source_code + "/blob/main/CODE_OF_CONDUCT.md",  # TODO: add this file
    },
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require=extras_require,
    keywords=keywords,
    classifiers=[
        "Development Status :: 3 - Alpha",
        # "Development Status :: 4 - Beta",
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
)
