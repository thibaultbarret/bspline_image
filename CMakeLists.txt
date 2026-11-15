# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.21)

# Extracted dependency archives (GoogleTest) get the extraction time as timestamp (CMake >= 3.24).
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" _bsi_src_dir)
file(REAL_PATH "${CMAKE_CURRENT_BINARY_DIR}" _bsi_bin_dir)
if(_bsi_src_dir STREQUAL _bsi_bin_dir)
  message(FATAL_ERROR
    "In-source builds are not allowed. Use a separate build directory, e.g.\n"
    "  cmake -S . -B build        (or: cmake --preset dev)\n"
    "and delete the CMakeCache.txt file and the CMakeFiles/ directory "
    "if they were created in the source tree.")
endif()

project(bspline_image VERSION 0.1.0 LANGUAGES CXX)

option(BSI_BUILD_TESTS "Build the unit tests" ${PROJECT_IS_TOP_LEVEL})
option(BSI_BUILD_EXAMPLES "Build the examples" ${PROJECT_IS_TOP_LEVEL})
option(BSI_BUILD_BENCH "Build the benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(BSI_BUILD_PYTHON "Build the Python bindings (nanobind)" OFF)
option(BSI_BUILD_DOCS "Build the documentation (Sphinx site and Doxygen API)" OFF)
option(BSI_WARNINGS_AS_ERRORS "Treat warnings as errors on internal targets" OFF)
option(BSI_SANITIZE "Enable AddressSanitizer and UBSan on internal targets" OFF)
option(BSI_COVERAGE "Enable coverage instrumentation on internal targets" OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(cmake/BsiWarnings.cmake)

# Sanitizer / coverage flags: internal targets only (carried by bsi_warnings).
if(BSI_SANITIZE AND NOT MSVC)
  target_compile_options(bsi_warnings INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(bsi_warnings INTERFACE -fsanitize=address,undefined)
endif()
if(BSI_COVERAGE AND NOT MSVC)
  target_compile_options(bsi_warnings INTERFACE --coverage)
  target_link_options(bsi_warnings INTERFACE --coverage)
endif()

add_library(bspline_image INTERFACE)
add_library(bsi::bspline_image ALIAS bspline_image)
target_compile_features(bspline_image INTERFACE cxx_std_20)
target_include_directories(bspline_image INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS bspline_image EXPORT bspline_imageTargets)
install(EXPORT bspline_imageTargets
  NAMESPACE bsi::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bspline_image)

configure_package_config_file(
  cmake/bspline_imageConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/bspline_imageConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bspline_image)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/bspline_imageConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/bspline_imageConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/bspline_imageConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bspline_image)

if(BSI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(BSI_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(BSI_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(BSI_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(BSI_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  configure_file(docs/Doxyfile.in ${CMAKE_BINARY_DIR}/docs/Doxyfile @ONLY)
  add_custom_target(docs-api
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/docs/Doxyfile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/docs
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)

  find_program(BSI_SPHINX_BUILD sphinx-build)
  if(NOT BSI_SPHINX_BUILD)
    message(FATAL_ERROR
      "sphinx-build not found: run pip install . -r docs/sphinx/requirements.txt in the active Python environment")
  endif()
  add_custom_target(docs
    COMMAND ${CMAKE_COMMAND} -E env BSI_DOCS_BUILD_DIR=${CMAKE_BINARY_DIR}/docs
            ${BSI_SPHINX_BUILD} -W --keep-going -b html
            -d ${CMAKE_BINARY_DIR}/docs/doctrees
            ${PROJECT_SOURCE_DIR}/docs/sphinx ${CMAKE_BINARY_DIR}/docs/site
    DEPENDS docs-api
    COMMENT "Building the Sphinx site"
    VERBATIM)
endif()
