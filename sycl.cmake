# SPDX-FileCopyrightText: 2021-2024 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

set(DEVICE_SOURCE_FILES device.cpp
        interfaces/sycl/Aux.cpp
        interfaces/sycl/Control.cpp
        interfaces/sycl/Copy.cpp
        interfaces/sycl/Events.cpp
        interfaces/sycl/Graphs.cpp
        interfaces/sycl/Memory.cpp
        interfaces/sycl/Streams.cpp
        interfaces/sycl/DeviceContext.cpp
        interfaces/sycl/DeviceCircularQueueBuffer.cpp
        interfaces/sycl/DeviceType.cpp
        algorithms/sycl/ArrayManip.cpp
        algorithms/sycl/BatchManip.cpp
        algorithms/sycl/Debugging.cpp
        algorithms/sycl/Reduction.cpp
        )

add_library(device SHARED ${DEVICE_SOURCE_FILES})

if ((${DEVICE_BACKEND} STREQUAL "acpp") OR (${DEVICE_BACKEND} STREQUAL "hipsycl"))
    if (DEVICE_ARCH MATCHES "sm_*")
        if (CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
            set(SYCL_USE_NVHPC_DEFAULT ON)
        else()
            set(SYCL_USE_NVHPC_DEFAULT OFF)
        endif()
        option(SYCL_USE_NVHPC "For Nvidia GPUs, use nvhpc instead of CUDA/nvcc." ${SYCL_USE_NVHPC_DEFAULT})
        if (SYCL_USE_NVHPC)
            # we assume that hipsycl was compiled with nvhpc compiler collection
            string(REPLACE sm_ cc NVCPP_ARCH ${DEVICE_ARCH})
            set(HIPSYCL_TARGETS "cuda-nvcxx:${NVCPP_ARCH}" CACHE STRING "" FORCE)
            set(ACPP_TARGETS "cuda-nvcxx:${NVCPP_ARCH}" CACHE STRING "" FORCE)
        else()
            set(HIPSYCL_TARGETS "cuda:${DEVICE_ARCH}" CACHE STRING "" FORCE)
            set(ACPP_TARGETS "cuda:${DEVICE_ARCH}" CACHE STRING "" FORCE)
            target_compile_options(device PRIVATE -Wno-unknown-cuda-version)
        endif()
    elseif(DEVICE_ARCH MATCHES "gfx*")
        set(HIPSYCL_TARGETS "hip:${DEVICE_ARCH}" CACHE STRING "" FORCE)
        set(ACPP_TARGETS "hip:${DEVICE_ARCH}" CACHE STRING "" FORCE)
    else()
        set(HIPSYCL_TARGETS "${DEVICE_BACKEND}:${DEVICE_ARCH}" CACHE STRING "" FORCE)
        set(ACPP_TARGETS "${DEVICE_BACKEND}:${DEVICE_ARCH}" CACHE STRING "" FORCE)
    endif()

    # hipSYCL or OpenSYCL is deemed to old now. If you _really_ should need it, look for the package right here.
    find_package(AdaptiveCpp REQUIRED)
    find_package(OpenMP REQUIRED)
    target_compile_options(device PRIVATE -Wno-unknown-cuda-version)
    target_link_libraries(device PUBLIC ${OpenMP_CXX_FLAGS})
    add_sycl_to_target(TARGET device SOURCES ${DEVICE_SOURCE_FILES})
else()
    find_package(DpcppFlags REQUIRED)
    target_link_libraries(device PRIVATE dpcpp::device_flags)
endif()
