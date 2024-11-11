# SPDX-FileCopyrightText: 2019-2024 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

#Add the kernel as a lib
cmake_minimum_required(VERSION 3.5)
project(subroutinesKernel)

set(TARGET_NAME kernel)

if (${DEVICE_BACKEND} STREQUAL "hip")
    include(hip.cmake)
elseif(${DEVICE_BACKEND} STREQUAL "cuda")
    include(cuda.cmake)
elseif((${DEVICE_BACKEND} STREQUAL "oneapi") OR (${DEVICE_BACKEND} STREQUAL "hipsycl"))
    include(sycl.cmake)
endif()
