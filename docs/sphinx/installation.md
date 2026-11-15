<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Installation

`bspline_image` is header-only and needs a C++20 compiler and CMake. It has no other
dependency.

## With FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(bspline_image
  GIT_REPOSITORY https://github.com/thibaultbarret/bspline_image.git
  GIT_TAG main)
FetchContent_MakeAvailable(bspline_image)
target_link_libraries(my_app PRIVATE bsi::bspline_image)
```

## From an installed copy

Install into a prefix, then use `find_package`:

```sh
cmake -S . -B build -G Ninja
cmake --install build --prefix <prefix>
```

```cmake
find_package(bspline_image 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE bsi::bspline_image)
```

## Building and testing the repository

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBSI_WARNINGS_AS_ERRORS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

In-source builds are refused: always configure into a separate directory.

CMake presets (`dev`, `release`, `python`, `docs`, `sanitize`; list them with
`cmake --list-presets`) provide ready-made configurations, also usable from IDEs:

```sh
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
```

Each preset builds in `build/<preset>`.

## Python module

The module is built with nanobind and is not published on PyPI. Install it from a
checkout, preferably in a virtual environment (it needs a C++20 compiler and CMake;
NumPy is required at run time):

```sh
python3 -m venv .venv && .venv/bin/pip install .
```

## Building this documentation

The site is built with Sphinx and the API reference with Doxygen (Doxygen must be
available from your system package manager). The documentation toolchain needs Python 3.12
or newer (Sphinx 9.1.1, `tomllib`). The Python module must be installed in the same
environment, because the pages execute examples that import it:

```sh
python3 -m venv .venv
.venv/bin/pip install . -r docs/sphinx/requirements.txt
PATH="$PWD/.venv/bin:$PATH" cmake --preset docs
PATH="$PWD/.venv/bin:$PATH" cmake --build --preset docs --target docs
```

The site is written to `build/docs/docs/site/index.html`, and the Doxygen API
reference to `build/docs/docs/extra/api/index.html`.

Local builds cache notebook outputs keyed on the notebook text only. After changing the
helper modules (`speckle.py`, `dic_problem.py`) or the library, remove the doctrees and
re-execute everything:

```sh
rm -rf build/docs/doctrees
PATH="$PWD/.venv/bin:$PATH" BSI_DOCS_EXECUTION=force cmake --build --preset docs --target docs
```
