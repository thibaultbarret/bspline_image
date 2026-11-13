# SPDX-License-Identifier: BSD-3-Clause
# Internal-only warning flags. Never propagated to consumers of bsi::bspline_image.

add_library(bsi_warnings INTERFACE)

if(MSVC)
  target_compile_options(bsi_warnings INTERFACE /W4 /permissive-)
  if(BSI_WARNINGS_AS_ERRORS)
    target_compile_options(bsi_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(bsi_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  if(BSI_WARNINGS_AS_ERRORS)
    target_compile_options(bsi_warnings INTERFACE -Werror)
  endif()
endif()
