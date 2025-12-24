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

target_compile_features(device PRIVATE cxx_std_17)

target_compile_definitions(device PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -DDEVICE_${BACKEND_UPPER_CASE}_LANG;
        >)

if (USE_GRAPH_CAPTURING)
  target_compile_definitions(device PRIVATE DEVICE_USE_GRAPH_CAPTURING)
endif()

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

find_package(CUDAToolkit REQUIRED)
target_link_libraries(device PUBLIC CUDA::cudart CUDA::cuda_driver)

if (ENABLE_PROFILING_MARKERS)
    target_link_libraries(device PUBLIC CUDA::nvtx3)
endif()
