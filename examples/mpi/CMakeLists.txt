# SPDX-FileCopyrightText: 2019 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.20)
project(MPI)


include(../cmake/common_options.cmake)


if(FORCE_INTEL_MPI)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
    find_package(IntelMPI REQUIRED)
else()
    find_package(MPI REQUIRED)
endif()


add_executable(mpi src/main.cpp)
add_subdirectory(../.. root)
target_link_libraries(mpi PUBLIC device ${MPI_CXX_LIBRARIES})


if(FORCE_INTEL_MPI)
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
        target_link_options(mpi PRIVATE ${MPI_CXX_LINK_FLAGS})
    else()
        set_target_properties(mpi PROPERTIES LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
    endif()
endif()


target_compile_options(mpi PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
target_compile_definitions(mpi PRIVATE DEVICE_${DEVICE_BACKEND}_LANG REAL_SIZE=${REAL_SIZE_IN_BYTES})
