# SPDX-FileCopyrightText: 2019 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.14)
project(device_submodule)

#set(CMAKE_CXX_CLANG_TIDY clang-tidy)
option(USE_GRAPH_CAPTURING "Enable graph capturing (if it is supported)" ON)

# check INTERFACE
if (NOT DEFINED DEVICE_BACKEND)
    message(FATAL_ERROR "DEVICE_BACKEND variable has not been provided into the submodule")
else()
    set(FOUND OFF)
    foreach(VARIANT cuda hip oneapi hipsycl acpp)
        if (${DEVICE_BACKEND} STREQUAL ${VARIANT})
            set(FOUND ON)
        endif()
    endforeach()
    if (NOT FOUND)
        message(FATAL_ERROR "DEVICE_BACKEND must be either cuda, hip, opeapi, acpp, or hipsycl. Given: ${DEVICE_BACKEND}")
    endif()
endif()

if (NOT DEFINED DEVICE_ARCH)
    message(FATAL_ERROR "DEVICE_ARCH is not defined. "
            "Supported for example: sm_60, sm_61, sm_70, sm_71, gfx906, gfx908, dg1, bdw, skl, Gen8, Gen9, Gen11, Gen12LP")
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(ENABLE_PROFILING_MARKERS "Enable profiling markers (if available)" OFF)

string(TOLOWER ${DEVICE_BACKEND} BACKEND_FOLDER)

# define device library
if (${DEVICE_BACKEND} STREQUAL "cuda")
    set(BACKEND_FOLDER "cuda")
    include(cuda.cmake)
elseif(${DEVICE_BACKEND} STREQUAL "hip")
    set(BACKEND_FOLDER "hip")
    include(hip.cmake)
elseif((${DEVICE_BACKEND} STREQUAL "oneapi") OR (${DEVICE_BACKEND} STREQUAL "hipsycl") OR (${DEVICE_BACKEND} STREQUAL "acpp"))
    set(BACKEND_FOLDER "sycl")
    include(sycl.cmake)
endif()

string(TOUPPER ${BACKEND_FOLDER} BACKEND_UPPER_CASE)

# common options
target_compile_features(device PRIVATE cxx_std_17)

if (USE_GRAPH_CAPTURING)
  target_compile_definitions(device PRIVATE DEVICE_USE_GRAPH_CAPTURING)
endif()
if (ENABLE_PROFILING_MARKERS)
  target_compile_definitions(device PRIVATE PROFILING_ENABLED)
endif()

target_compile_definitions(device PRIVATE DEVICE_LANG_${BACKEND_UPPER_CASE})

if (LOG_LEVEL_MASTER)
  target_compile_definitions(device PRIVATE LOG_LEVEL=${LOG_LEVEL_MASTER})
endif()

target_include_directories(device PUBLIC .)

if (DEFINED DEVICE_SUBMODULES)
    target_include_directories(device PRIVATE ${DEVICE_SUBMODULES})
else()
    target_include_directories(device PRIVATE submodules)
endif()
