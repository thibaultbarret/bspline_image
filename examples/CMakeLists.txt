# SPDX-License-Identifier: BSD-3-Clause
foreach(example synthetic_image dic_visit dic_parallel_subsets)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE bsi::bspline_image bsi_warnings)
endforeach()

target_link_libraries(dic_parallel_subsets PRIVATE Threads::Threads)

if(BSI_BUILD_TESTS)
  add_test(NAME example_dic_parallel_subsets COMMAND dic_parallel_subsets)
endif()
