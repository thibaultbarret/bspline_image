# SPDX-License-Identifier: BSD-3-Clause
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE BSI_NANOBIND_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
find_package(nanobind CONFIG HINTS "${BSI_NANOBIND_DIR}")
if(NOT nanobind_FOUND)
  include(FetchContent)
  FetchContent_Declare(nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG v2.4.0
    GIT_SUBMODULES "ext/robin_map")
  FetchContent_MakeAvailable(nanobind)
endif()

# Strict warnings (bsi_warnings) are intentionally not applied to the module.
nanobind_add_module(_core NB_STATIC NOMINSIZE src/bindings.cpp)
target_link_libraries(_core PRIVATE bsi::bspline_image bsi_native)

nanobind_add_stub(_core_stub
  MODULE _core
  OUTPUT _core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_core>
  DEPENDS _core)

install(TARGETS _core LIBRARY DESTINATION bspline_image COMPONENT python)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_core.pyi
  DESTINATION bspline_image COMPONENT python)
