cmake_minimum_required(VERSION 3.5)
project(jsolver)
set(CMAKE_CXX_STANDARD 17)

option(WITH_TIDY "turn on clang-tidy" OFF)

if (WITH_TIDY)
    set(CMAKE_CXX_CLANG_TIDY clang-tidy;-header-filter=.*;)
endif()

option(WITH_TESTS "build with unit test" OFF)

#8 for double; 4 for float
set(REAL_SIZE_IN_BYTES "8" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE_IN_BYTES PROPERTY STRINGS "8" "4")

set(DEVICE_BACKEND "CUDA" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "CUDA" "HIP")

set(SM "sm_60" CACHE STRING "model of Streaming Multiprocessor")
set(SM_OPTIONS "sm_20" "sm_30" "sm_50" "sm_60" "sm_70" "gfx906")
set_property(CACHE SM PROPERTY STRINGS ${SM_OPTIONS})

option(WITH_MPI "build with MPI support" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (WITH_MPI)
    find_package(MPI REQUIRED)
    set(__MPI_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
    set(__MPI_LIBS ${MPI_CXX_LIBRARIES})
    set(__MPI_DEFINITIONS USE_MPI)
endif()

find_package(yaml-cpp REQUIRED)

set(DEVICE_SUB_ARCH ${SM})

add_subdirectory(root)
add_subdirectory(src/gpu/kernels)

add_executable(solver src/main.cpp
                      src/host/subroutines.cpp
                      src/host/solver.cpp
                      src/gpu/solver.cpp
                      src/matrix_manip.cpp)


target_include_directories(solver PUBLIC src
                                         root
                                         ${__MPI_INCLUDES}
                                         ${YAML_CPP_INCLUDE_DIR})

target_link_libraries(solver PUBLIC device
                                    kernel
                                    ${__MPI_LIBS}
                                    ${YAML_CPP_LIBRARIES})

target_compile_definitions(solver PUBLIC REAL_SIZE=${REAL_SIZE_IN_BYTES}
                                         ${__MPI_DEFINITIONS})

configure_file(${PROJECT_SOURCE_DIR}/input/example.yaml
               ${PROJECT_BINARY_DIR}/config.yaml
               COPYONLY)

if (WITH_TESTS)
    find_package(GTest REQUIRED)
    find_package(PkgConfig)
    pkg_check_modules(GMOCK "gmock" REQUIRED)

    add_executable(tests tests/driver.cpp
                         tests/host/vectors_tests.cpp
                         tests/host/matrix_init.cpp
                         tests/host/matrix_subroutines.cpp
                         src/host/subroutines.cpp
                         src/matrix_manip.cpp)

    target_include_directories(tests PRIVATE ${GTEST_INCLUDE_DIR}
                                             ${GMOCK_INCLUDE_DIR}
                                             ${__MPI_INCLUDES}
                                             ${CMAKE_SOURCE_DIR}/src)

    target_link_libraries(tests PRIVATE ${GTEST_LIBRARIES}
                                        ${GMOCK_LIBRARIES}
                                        ${__MPI_LIBS})

    target_compile_definitions(tests PRIVATE REAL_SIZE=${REAL_SIZE_IN_BYTES}
                                             ${__MPI_DEFINITIONS})
endif()
