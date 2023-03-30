cmake_minimum_required(VERSION 3.5)
project(examples)

#set(CMAKE_CXX_CLANG_TIDY clang-tidy)

set(SM "sm_60" CACHE STRING "model of Nvidia Streaming Multiprocessor")
set(SM_OPTIONS "sm_50" "sm_60" "sm_70" "sm_71" "sm_75" "sm_80" "sm_86"
      "gfx906" "gfx908 "
      "dg1" "bdw" "skl" "Gen8" "Gen9" "Gen11" "Gen12LP")
set_property(CACHE SM PROPERTY STRINGS ${SM_OPTIONS})

set(DEVICE_ARCH ${SM})

set(DEVICE_BACKEND "cuda" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "cuda" "hip" "oneapi" "opensycl")

set(REAL_SIZE_IN_BYTES "8" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE_IN_BYTES PROPERTY STRINGS "8" "4")

add_subdirectory(root)

find_package(GTest REQUIRED)

add_executable(tests main.cpp reductions.cpp array_manip.cpp memory.cpp)
target_link_libraries(tests device ${GTEST_BOTH_LIBRARIES})
target_include_directories(tests PRIVATE root ${GTEST_INCLUDE_DIR})
target_compile_definitions(tests PRIVATE REAL_SIZE=${REAL_SIZE_IN_BYTES})

enable_testing()
add_test(NAME all_tests COMMAND tests)
