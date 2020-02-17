cmake_minimum_required(VERSION 3.5)
project(examples)

set(INTERFACE "DUMMY" CACHE STRING "type of an interface")
set_property(CACHE INTERFACE PROPERTY STRINGS "DUMMY" "CUDA")

set(REAL_SIZE "8" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE PROPERTY STRINGS "8" "4")

add_executable(example main.cpp)
target_include_directories(example PUBLIC root)

add_subdirectory(root)
target_link_libraries(example device)