# SPDX-License-Identifier: BSD-3-Clause
# Standalone project validating the installed bspline_image package.
cmake_minimum_required(VERSION 3.21)
project(bsi_install_consumer LANGUAGES CXX)

find_package(bspline_image 0.1 REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE bsi::bspline_image)
target_compile_features(consumer PRIVATE cxx_std_20)
