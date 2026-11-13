# SPDX-License-Identifier: BSD-3-Clause
include(FetchContent)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

file(GLOB BSI_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
add_executable(bsi_tests ${BSI_TEST_SOURCES})
target_include_directories(bsi_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bsi_tests PRIVATE bsi::bspline_image bsi_warnings GTest::gtest_main)
gtest_discover_tests(bsi_tests)
