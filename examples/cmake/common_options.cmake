# SPDX-FileCopyrightText: 2021-2024 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

option(WITH_TIDY "turn on clang-tidy" OFF)

if (WITH_TIDY)
    set(CMAKE_CXX_CLANG_TIDY clang-tidy;-header-filter=.*;)
endif()

#8 for double; 4 for float
set(REAL_SIZE_IN_BYTES "8" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE_IN_BYTES PROPERTY STRINGS "8" "4")

set(DEVICE_BACKEND "cuda" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "cuda" "hip" "oneapi" "hipsycl")

set(DEVICE_ARCH "sm_60" CACHE STRING "model of Streaming Multiprocessor")
set(DEVICE_ARCH_OPTIONS "sm_50" "sm_60" "sm_70" "sm_71" "sm_75" "sm_80" "sm_86"
    "gfx906" "gfx908 "
    "dg1" "bdw" "skl" "Gen8" "Gen9" "Gen11" "Gen12LP")
set_property(CACHE DEVICE_ARCH PROPERTY STRINGS ${DEVICE_ARCH_OPTIONS})

