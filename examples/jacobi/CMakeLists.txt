# SPDX-FileCopyrightText: 2019 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.20)
project(jsolver)

include(../cmake/common_options.cmake)
option(WITH_TESTS "build with unit test" OFF)
option(WITH_MPI "build with MPI support" ON)

find_package(yaml-cpp REQUIRED 0.7.0)
add_subdirectory(../.. root)
add_subdirectory(src/gpu/kernels)

if (WITH_MPI)
    # workaround for Intel MPI: find mpi does not set all variables correctly,
    # therefore we need some manual configuration
    if(FORCE_INTEL_MPI)
        # the submodule does not search within the cmake folder,
        # thus we are extending the module path
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
        find_package(IntelMPI REQUIRED)
        set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

        # note: __MPI_LINK_FLAGS relevant only for IntelMPI
        set(__MPI_LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
    else()
        find_package(MPI REQUIRED)
    endif()

    set(__MPI_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
    set(__MPI_LIBS ${MPI_CXX_LIBRARIES})
    set(__MPI_COMPILE_FLAGS ${MPI_CXX_COMPILE_OPTIONS})
    set(__MPI_DEFINITIONS ${MPI_CXX_COMPILE_DEFINITIONS} "USE_MPI")
endif()

add_executable(solver src/main.cpp
                      src/host/subroutines.cpp
                      src/host/solver.cpp
                      src/gpu/solver.cpp
                      src/matrix_manip.cpp)

if (${DEVICE_BACKEND} STREQUAL "hipsycl")
    find_package(OpenMP REQUIRED)
    target_link_libraries(solver PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(device PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(kernel PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(solver PUBLIC src
                                         ${__MPI_INCLUDES})

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
    target_link_options(solver PUBLIC ${__MPI_LINK_FLAGS})
else()
    set_target_properties(solver PROPERTIES LINK_FLAGS "${__MPI_LINK_FLAGS}")
endif()


target_link_libraries(solver PUBLIC device
                                    kernel
                                    yaml-cpp
                                    ${__MPI_LIBS})



target_compile_definitions(solver PUBLIC REAL_SIZE=${REAL_SIZE_IN_BYTES}
                                         ${__MPI_DEFINITIONS})

target_compile_options(solver PUBLIC -Wall -Wextra -pedantic -std=c++17 -O3 ${__MPI_COMPILE_FLAGS})

configure_file(${PROJECT_SOURCE_DIR}/input/example.yaml
               ${PROJECT_BINARY_DIR}/config.yaml
               COPYONLY)

if (WITH_TESTS)
    find_package(GTest REQUIRED)

    add_executable(tests tests/driver.cpp
                         tests/host/vectors_tests.cpp
                         tests/host/matrix_init.cpp
                         tests/host/matrix_subroutines.cpp
                         src/host/subroutines.cpp
                         tests/gpu/subroutines_tests.cpp
                         src/matrix_manip.cpp)

    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
        target_link_options(tests PUBLIC ${__MPI_LINK_FLAGS})
    endif()

    target_include_directories(tests PRIVATE ${GTEST_INCLUDE_DIR}
                                             ${__MPI_INCLUDES}
                                             ${CMAKE_SOURCE_DIR}/src)

    target_link_libraries(tests PRIVATE device
                                        kernel
                                        ${GTEST_BOTH_LIBRARIES}
                                        ${__MPI_LIBS})

    target_compile_definitions(tests PRIVATE REAL_SIZE=${REAL_SIZE_IN_BYTES}
                                     ${__MPI_DEFINITIONS})

    target_compile_options(tests PRIVATE -std=c++17 ${__MPI_COMPILE_FLAGS})

endif()
