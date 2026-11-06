# SPDX-FileCopyrightText: 2020 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

if(NOT DEFINED ROCM_PATH)
    if (NOT DEFINED ENV{ROCM_PATH})
        # default location
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
    else()
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
    endif()
endif()

# ensure that we have set HIP_PATH
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

# set the CMAKE_MODULE_PATH for the helper cmake files from HIP
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" "${HIP_PATH}/lib/cmake/hip" "${ROCM_PATH}/cmake" ${CMAKE_MODULE_PATH})

# TODO: switch to CMake HIP support once CMake 3.21 is the minimum version
find_package(HIP REQUIRED)

# Can set different FLAGS for the compilers via HCC_OPTIONS and NVCC_OPTIONS keywords;
# options for both via HIPCC_OPTIONS
# Set the flags here, use them later
# Only need NVCC at the time no AMD system to deploy to
set(DEVICE_HIPCC)
set(IS_NVCC_PLATFORM OFF)
if (DEFINED ENV{HIP_PLATFORM})
    if ($ENV{HIP_PLATFORM} STREQUAL "nvidia")
        set(IS_NVCC_PLATFORM ON)
    endif()
endif()

if (IS_NVCC_PLATFORM)
    set(DEVICE_NVCC -arch=${DEVICE_ARCH};
                    -dc;
                    --expt-relaxed-constexpr)
else()
    set(DEVICE_HIPCC -std=c++17;
                     -O3;
                     --offload-arch=${DEVICE_ARCH})
endif()

if (DEVICE_KERNEL_INFOPRINT)
    set(DEVICE_HIPCC ${DEVICE_HIPCC} -Rpass-analysis=kernel-resource-usage)
endif()
if (DEVICE_KERNEL_SAVETEMPS)
    set(DEVICE_HIPCC ${DEVICE_HIPCC} --save-temps)
endif()


# Streams on HSA queues of our own (see interfaces/hip/hsaqueues/HsaQueues.h): a module of the HIP
# interface that only programs enable that ask for it (AbstractAPI::enableHsaQueues()); without it,
# a stub that never enables them
if (IS_NVCC_PLATFORM)
    set(DEVICE_HIP_HSA_QUEUES OFF)
else()
    option(DEVICE_HIP_HSA_QUEUES "Build the streams on HSA queues of our own (AMD GPUs)" ON)
endif()
if (DEVICE_HIP_HSA_QUEUES)
    set(DEVICE_HSA_QUEUE_SOURCES interfaces/hip/hsaqueues/Arch.cpp
                                 interfaces/hip/hsaqueues/Common.cpp
                                 interfaces/hip/hsaqueues/Kernels.cpp
                                 interfaces/hip/hsaqueues/Metadata.cpp
                                 interfaces/hip/hsaqueues/HsaQueues.cpp
                                 interfaces/hip/hsaqueues/Program.cpp
                                 interfaces/hip/hsaqueues/Runtime.cpp
                                 interfaces/hip/hsaqueues/Trace.cpp)
else()
    set(DEVICE_HSA_QUEUE_SOURCES interfaces/hip/hsaqueues/Disabled.cpp)
endif()

set(DEVICE_SOURCE_FILES device.cpp
                        interfaces/hip/Control.cpp
                        interfaces/hip/Copy.cpp
                        interfaces/hip/Events.cpp
                        interfaces/hip/Internals.cpp
                        interfaces/hip/Memory.cpp
                        interfaces/hip/Streams.cpp
                        interfaces/hip/Graphs.cpp
                        ${DEVICE_HSA_QUEUE_SOURCES}
                        algorithms/cudahip/ArrayManip.cpp
                        algorithms/cudahip/BatchManip.cpp
                        algorithms/cudahip/Debugging.cpp
                        algorithms/cudahip/Reduction.cpp)

set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_PATH} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")


set_source_files_properties(${DEVICE_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_reset_flags()
hip_add_library(device ${DEVICE_LIBTYPE} ${DEVICE_SOURCE_FILES}
                       HIPCC_OPTIONS ${DEVICE_HIPCC}
                       NVCC_OPTIONS ${DEVICE_NVCC})

set_property(TARGET device PROPERTY HIP_ARCHITECTURES OFF)

if (IS_NVCC_PLATFORM)
    set_target_properties(device PROPERTIES LINKER_LANGUAGE HIP)
    target_link_options(device PRIVATE -arch=${DEVICE_ARCH})
else()
    target_link_libraries(device PUBLIC ${HIP_PATH}/lib/libamdhip64.so)
endif()
if (DEVICE_HIP_HSA_QUEUES)
    target_link_libraries(device PRIVATE ${HIP_PATH}/lib/libhsa-runtime64.so ${HIP_PATH}/lib/libamd_comgr.so)
endif()

# The launch hook for HSA queues, to be compiled into the program itself (see
# interfaces/hip/hsaqueues/Hook.cpp): link this target to the executable. Without HSA queues, it
# brings nothing.
add_library(device-hsa-queue-hook INTERFACE)
if (DEVICE_HIP_HSA_QUEUES)
    target_sources(device-hsa-queue-hook INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/hip/hsaqueues/Hook.cpp)
    target_include_directories(device-hsa-queue-hook INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/hip/hsaqueues ${HIP_PATH}/include)
    target_compile_definitions(device-hsa-queue-hook INTERFACE __HIP_PLATFORM_AMD__)
    target_link_libraries(device-hsa-queue-hook INTERFACE device ${CMAKE_DL_LIBS})
endif()

# the test and the probes of HSA queues (see tests/hsaqueues/README.md)
if (DEVICE_HIP_HSA_QUEUES)
    option(DEVICE_HSA_QUEUE_TESTS "Build the test and the probes of HSA queues" OFF)
    if (DEVICE_HSA_QUEUE_TESTS)
        include(${CMAKE_CURRENT_SOURCE_DIR}/tests/hsaqueues/hsaqueues.cmake)
    endif()
endif()

if (ENABLE_PROFILING_MARKERS)
    # cf. https://github.com/ROCm/rocprofiler/blob/amd-master/tests-v2/featuretests/tracer/CMakeLists.txt
    find_library(ROCTX_LIBRARY NAMES roctx64 HINTS ${ROCM_PATH}/lib)
    target_link_libraries(device PRIVATE ${ROCTX_LIBRARY})
endif()
