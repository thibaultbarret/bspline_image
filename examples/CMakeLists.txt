# SPDX-License-Identifier: BSD-3-Clause
foreach(example synthetic_image dic_visit)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE bsi::bspline_image bsi_warnings)
endforeach()
