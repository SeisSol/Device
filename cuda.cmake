# SPDX-FileCopyrightText: 2020-2024 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

add_library(device SHARED device.cpp
                 interfaces/cuda/Control.cu
                 interfaces/cuda/Aux.cu
                 interfaces/cuda/Memory.cu
                 interfaces/cuda/Events.cu
                 interfaces/cuda/Copy.cu
                 interfaces/cuda/Streams.cu
                 interfaces/cuda/Graphs.cu
                 interfaces/cuda/Internals.cu
                 algorithms/cuda/ArrayManip.cu
                 algorithms/cuda/BatchManip.cu
                 algorithms/cuda/Debugging.cu
                 algorithms/cuda/Reduction.cpp)

set_target_properties(device PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_source_files_properties(device.cpp
                            algorithms/cuda/Reduction.cpp
                            PROPERTIES LANGUAGE CUDA)


string(REPLACE "sm_" "" CUDA_DEVICE_ARCH "${DEVICE_ARCH}")
set_target_properties(device PROPERTIES CUDA_ARCHITECTURES "${CUDA_DEVICE_ARCH}")

target_compile_features(device PRIVATE cxx_std_17)

target_compile_definitions(device PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -DDEVICE_${BACKEND_UPPER_CASE}_LANG;
        -DREAL_SIZE=${REAL_SIZE_IN_BYTES}
        >)

if (USE_GRAPH_CAPTURING)
  target_compile_definitions(device PRIVATE DEVICE_USE_GRAPH_CAPTURING)
endif()

target_compile_options(device PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -Xptxas -v;
        --expt-relaxed-constexpr;
        >)

if (EXTRA_DEVICE_FLAGS)
    target_compile_options(device PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            ${EXTRA_DEVICE_FLAGS}
            >)
endif()

find_package(CUDAToolkit REQUIRED)
target_link_libraries(device PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvToolsExt)

