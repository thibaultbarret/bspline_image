# SPDX-License-Identifier: BSD-3-Clause
add_executable(bsi_bench bench_main.cpp)
target_link_libraries(bsi_bench PRIVATE bsi::bspline_image bsi_warnings)
