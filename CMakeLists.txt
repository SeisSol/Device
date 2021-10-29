cmake_minimum_required(VERSION 3.5)
project(device_submodule)

#set(CMAKE_CXX_CLANG_TIDY clang-tidy)

# check INTERFACE
if (NOT DEFINED DEVICE_BACKEND)
    message(FATAL_ERROR "DEVICE_BACKEND variable has not been provided into the submodule")
else()
    set(FOUND OFF)
    foreach(VARIANT cuda hip oneapi hipsycl)
        if (${DEVICE_BACKEND} STREQUAL ${VARIANT})
            set(FOUND ON)
        endif()
    endforeach()
    if (NOT FOUND)
        message(FATAL_ERROR "DEVICE_BACKEND must be either cuda, hip, opeapi, or hipsycl. Given: ${DEVICE_BACKEND}")
    endif()
endif()

if (NOT DEFINED DEVICE_ARCH)
    message(FATAL_ERROR "DEVICE_ARCH is not defined. "
            "Supported for example: sm_60, sm_61, sm_70, sm_71, gfx906, gfx908, dg1, bdw, skl, Gen8, Gen9, Gen11, Gen12LP")
endif()


#check REAL_SIZE
if (NOT DEFINED REAL_SIZE_IN_BYTES)
    message(FATAL_ERROR "REAL_SIZE_IN_BYTES variable has not been provided into the submodule")
else()
    set(FOUND OFF)
    foreach(VARIANT 4 8)
        if (${REAL_SIZE_IN_BYTES} EQUAL ${VARIANT})
            set(FOUND ON)
        endif()
    endforeach()
    if (NOT FOUND)
        message(FATAL_ERROR "REAL_SIZE_IN_BYTES must be either 4 or 8. Given: ${REAL_SIZE}")
    endif()
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

string(TOLOWER ${DEVICE_BACKEND} BACKEND_FOLDER)
string(TOUPPER ${DEVICE_BACKEND} BACKEND_UPPER_CASE)

if (${DEVICE_BACKEND} STREQUAL "cuda")
    set(BACKEND_FOLDER "cuda")
    include(nvidia.cmake)
elseif(${DEVICE_BACKEND} STREQUAL "hip")
    set(BACKEND_FOLDER "hip")
    include(amd.cmake)
elseif((${DEVICE_BACKEND} STREQUAL "oenapi") OR (${DEVICE_BACKEND} STREQUAL "hipsycl"))
    set(BACKEND_FOLDER "sycl")
    include(sycl.cmake)
endif()

target_compile_definitions(device PRIVATE DEVICE_${BACKEND_UPPER_CASE}_LANG REAL_SIZE=${REAL_SIZE_IN_BYTES})

target_include_directories(device PRIVATE .
                                          interfaces/${BACKEND_FOLDER}
                                          interfaces/common
                                          algorithms/${BACKEND_FOLDER}
                                          submodules)

target_compile_options(device PRIVATE -fPIC)


