cmake_minimum_required(VERSION 3.5)
project(examples)

set(CMAKE_CXX_CLANG_TIDY clang-tidy)

set(SM "sm_60" CACHE STRING "model of Nvidia Streaming Multiprocessor")
set(SM_OPTIONS "sm_20" "sm_30" "sm_50" "sm_60" "sm_70")
set_property(CACHE SM PROPERTY STRINGS ${SM_OPTIONS})

set(COMPUTE_SUB_ARCH ${SM})

set(DEVICE_BACKEND "CUDA" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "DUMMY" "CUDA")

set(REAL_SIZE_IN_BYTES "8" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE_IN_BYTES PROPERTY STRINGS "8" "4")

add_subdirectory(root)

set(SEISSOL_SRC ../../../src)

add_executable(basic basic.cpp)
target_include_directories(basic PUBLIC root ${SEISSOL_SRC})
target_link_libraries(basic device)

find_package(OpenBLAS QUIET)
if (OpenBLAS_FOUND)
    add_executable(gemm gemm.cpp)
    target_include_directories(gemm PUBLIC root ${OpenBLAS_INCLUDE_DIRS} ${SEISSOL_SRC})
    target_link_libraries(gemm device ${OpenBLAS_LIBRARIES})
    target_compile_definitions(gemm PRIVATE REAL_SIZE=${REAL_SIZE_IN_BYTES})

    add_executable(streams streams.cpp)
    target_include_directories(streams PUBLIC root ${OpenBLAS_INCLUDE_DIRS} ${SEISSOL_SRC})
    target_link_libraries(streams device ${OpenBLAS_LIBRARIES})
    target_compile_definitions(streams PRIVATE REAL_SIZE=${REAL_SIZE_IN_BYTES} )

else()
    message(STATUS "OpenBLAS has not been found. Cannot build GEMM example")
endif()

