# SPDX-FileCopyrightText: 2020 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

enable_language(HIP)
set(CMAKE_HIP_STANDARD 17)

add_library(device ${DEVICE_LIBTYPE} device.cpp
                        interfaces/hip/Control.cpp
                        interfaces/hip/Copy.cpp
                        interfaces/hip/Events.cpp
                        interfaces/hip/Internals.cpp
                        interfaces/hip/Memory.cpp
                        interfaces/hip/Streams.cpp
                        interfaces/hip/Graphs.cpp
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

set_target_properties(device PROPERTIES HIP_ARCHITECTURES "${DEVICE_ARCH}")

target_compile_features(device PRIVATE cxx_std_17)

target_compile_definitions(device PRIVATE $<$<COMPILE_LANGUAGE:HIP>:
        -DDEVICE_${BACKEND_UPPER_CASE}_LANG;
        >)

if (USE_GRAPH_CAPTURING)
  target_compile_definitions(device PRIVATE DEVICE_USE_GRAPH_CAPTURING)
endif()
if (DEVICE_KERNEL_INFOPRINT)
    target_compile_options(device PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-Rpass-analysis=kernel-resource-usage>)
endif()
if (DEVICE_KERNEL_SAVETEMPS)
    target_compile_options(device PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--save-temps>)
endif()

if(NOT DEFINED ROCM_PATH)
    if (NOT DEFINED ENV{ROCM_PATH})
        # default location
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
    else()
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
    endif()
endif()

# set the CMAKE_MODULE_PATH for the helper cmake files from HIP
set(CMAKE_MODULE_PATH "${ROCM_PATH}/cmake" ${CMAKE_MODULE_PATH})

if (ENABLE_PROFILING_MARKERS)
    # cf. https://github.com/ROCm/rocprofiler/blob/amd-master/tests-v2/featuretests/tracer/CMakeLists.txt
    find_library(ROCTX_LIBRARY NAMES roctx64 HINTS ${ROCM_PATH}/lib)
    target_link_libraries(device PRIVATE ${ROCTX_LIBRARY})
endif()
