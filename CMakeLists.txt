# TODO: comment 

cmake_minimum_required(VERSION 3.5)
project(mock_project)


find_package(CUDA REQUIRED)

set(HARDWARE_DEFINITIONS "REAL_SIZE=8" "ALIGNMENT=32")
set(ACCELERATOR_LIB "cuda")

add_executable(${CMAKE_PROJECT_NAME} api_testing.cpp)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${ACCELERATOR_LIB}/src)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ${HARDWARE_DEFINITIONS})

add_subdirectory(${ACCELERATOR_LIB}) # compiles 'custom_blas' library
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE custom_blas)

message(S"libname: ${CUSTOM_BLAS_LIBRARY}")
