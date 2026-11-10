# SPDX-FileCopyrightText: 2020 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

add_library(device ${DEVICE_LIBTYPE} device.cpp
                 interfaces/cuda/Control.cu
                 interfaces/cuda/Memory.cu
                 interfaces/cuda/Events.cu
                 interfaces/cuda/Copy.cu
                 interfaces/cuda/Streams.cu
                 interfaces/cuda/Graphs.cu
                 interfaces/cuda/Internals.cu
                 interfaces/cuda/Driver.cu
                 algorithms/cudahip/ArrayManip.cpp
                 algorithms/cudahip/BatchManip.cpp
                 algorithms/cudahip/Debugging.cpp
                 algorithms/cudahip/Reduction.cpp)

set_target_properties(device PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_source_files_properties(device.cpp
                                algorithms/cudahip/ArrayManip.cpp
                                algorithms/cudahip/BatchManip.cpp
                                algorithms/cudahip/Debugging.cpp
                                algorithms/cudahip/Reduction.cpp
                            PROPERTIES LANGUAGE CUDA)


string(REPLACE "sm_" "" CUDA_DEVICE_ARCH "${DEVICE_ARCH}")
set_target_properties(device PROPERTIES CUDA_ARCHITECTURES "${CUDA_DEVICE_ARCH}")

target_compile_options(device PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr;
        >)

if (DEVICE_KERNEL_INFOPRINT)
        target_compile_options(device PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                -Xptxas -v;
                >)
endif()

if (DEVICE_KERNEL_SAVETEMPS)
        target_compile_options(device PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                --keep;
                >)
endif()

if (EXTRA_DEVICE_FLAGS)
    target_compile_options(device PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            ${EXTRA_DEVICE_FLAGS}
            >)
endif()

# CUDA 11.7 has the variants of the driver functions that interfaces/cuda/Driver.h looks up
find_package(CUDAToolkit 11.7 REQUIRED)
# The driver functions are looked up through the runtime; linking the driver library (libcuda)
# would keep any program linked with the library from starting where there is no driver.
target_link_libraries(device PUBLIC CUDA::cudart)

if (ENABLE_PROFILING_MARKERS)
    target_link_libraries(device PUBLIC CUDA::nvtx3)
endif()
