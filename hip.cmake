# SPDX-FileCopyrightText: 2020-2024 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

# ensure that we have set HIP_PATH
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        if (NOT DEFINED ENV{ROCM_PATH})
            # default location
            set(HIP_PATH "/opt/rocm" CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

# set the CMAKE_MODULE_PATH for the helper cmake files from HIP
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" "${HIP_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})

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
                    --expt-relaxed-constexpr;
                    -DCUDA_UNDERHOOD)
else()
    set(DEVICE_HIPCC -std=c++17;
                     -O3;
                     --offload-arch=${DEVICE_ARCH};
                     -DDEVICE_${BACKEND_UPPER_CASE}_LANG)
endif()


set(DEVICE_SOURCE_FILES device.cpp
                        interfaces/hip/Aux.cpp
                        interfaces/hip/Control.cpp
                        interfaces/hip/Copy.cpp
                        interfaces/hip/Events.cpp
                        interfaces/hip/Internals.cpp
                        interfaces/hip/Memory.cpp
                        interfaces/hip/Streams.cpp
                        interfaces/hip/Graphs.cpp
                        algorithms/hip/ArrayManip.cpp
                        algorithms/hip/BatchManip.cpp
                        algorithms/hip/Debugging.cpp
                        algorithms/hip/Reduction.cpp)

set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_PATH} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")


set_source_files_properties(${DEVICE_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_reset_flags()
hip_add_library(device SHARED ${DEVICE_SOURCE_FILES}
                       HIPCC_OPTIONS ${DEVICE_HIPCC}
                       NVCC_OPTIONS ${DEVICE_NVCC})

set_property(TARGET device PROPERTY HIP_ARCHITECTURES OFF)

if (IS_NVCC_PLATFORM)
    set_target_properties(device PROPERTIES LINKER_LANGUAGE HIP)
    target_link_options(device PRIVATE -arch=${DEVICE_ARCH})
else()
    target_link_libraries(device PUBLIC ${HIP_PATH}/lib/libamdhip64.so)
endif()

