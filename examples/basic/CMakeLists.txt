cmake_minimum_required(VERSION 3.5)
project(examples)

option(WITH_TIDY "turn on clang-tidy" OFF)

if (WITH_TIDY)
    set(CMAKE_CXX_CLANG_TIDY clang-tidy;-header-filter=.*;)
endif()

set(SM "sm_60" CACHE STRING "model of Streaming Multiprocessor")
set(SM_OPTIONS "sm_20" "sm_30" "sm_50" "sm_60" "sm_70" "gfx906" "bdw" "skl" "kbl" "cfl" "bxt" "glk" "icllp" "lkf" "ehl" "tgllp" "rkl" "adls" "dg1" "Gen8" "Gen9" "Gen11" "Gen12LP")
set_property(CACHE SM PROPERTY STRINGS ${SM_OPTIONS})

set(DEVICE_SUB_ARCH ${SM})

set(DEVICE_BACKEND "CUDA" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "CUDA" "HIP" "ONEAPI")

set(REAL_SIZE_IN_BYTES "8" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE_IN_BYTES PROPERTY STRINGS "8" "4")

add_subdirectory(root)
add_executable(basic basic.cpp)
target_include_directories(basic PUBLIC root)
target_link_libraries(basic device)
target_compile_definitions(basic PRIVATE REAL_SIZE=${REAL_SIZE_IN_BYTES})
