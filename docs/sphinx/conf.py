# SPDX-License-Identifier: BSD-3-Clause
"""Sphinx configuration for the bspline_image documentation."""

import os
import tomllib
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parent.parent
BUILD_DIR = Path(os.environ.get("BSI_DOCS_BUILD_DIR", REPO_ROOT / "build" / "docs"))

with open(REPO_ROOT / "pyproject.toml", "rb") as pyproject:
    release = tomllib.load(pyproject)["project"]["version"]

project = "bspline_image"
copyright = "2026, Thibault BARRET"

extensions = [
    "myst_nb",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.doxylink",
]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]

html_theme = "sphinx_rtd_theme"
# The Doxygen HTML (built by the docs-api target) is published under api/.
html_extra_path = [str(BUILD_DIR / "extra")]

bibtex_bibfiles = ["references.bib"]

doxylink = {"bsi": (str(BUILD_DIR / "bsi.tag"), "api")}

nb_execution_mode = os.environ.get("BSI_DOCS_EXECUTION", "cache")
nb_execution_cache_path = str(BUILD_DIR / "jupyter_cache")
nb_execution_raise_on_error = True
nb_execution_timeout = 120

exclude_patterns = ["**.ipynb_checkpoints"]
