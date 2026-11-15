# SPDX-License-Identifier: BSD-3-Clause
# Internal-only CPU-specific optimisation flag. Never attached to bsi::bspline_image
# and never exported: binaries built with it only run on compatible CPUs.

include(CheckCXXCompilerFlag)

add_library(bsi_native INTERFACE)

if(BSI_NATIVE)
  if(MSVC)
    message(STATUS "BSI_NATIVE ignored with MSVC")
  else()
    check_cxx_compiler_flag(-march=native BSI_HAS_MARCH_NATIVE)
    if(BSI_HAS_MARCH_NATIVE)
      set(_bsi_native_flag -march=native)
    else()
      check_cxx_compiler_flag(-mcpu=native BSI_HAS_MCPU_NATIVE)
      if(BSI_HAS_MCPU_NATIVE)
        set(_bsi_native_flag -mcpu=native)
      endif()
    endif()

    if(_bsi_native_flag)
      target_compile_options(bsi_native INTERFACE ${_bsi_native_flag})
      message(STATUS
        "BSI_NATIVE: ${_bsi_native_flag} (binaries only run on CPUs compatible with this machine)")
    else()
      message(WARNING "BSI_NATIVE requested but neither -march=native nor -mcpu=native is supported")
    endif()
  endif()
endif()
