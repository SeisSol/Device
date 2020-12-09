cmake_minimum_required(VERSION 3.5)
project(examples)

#set(CMAKE_CXX_CLANG_TIDY clang-tidy)

set(SM "sm_60" CACHE STRING "model of Nvidia Streaming Multiprocessor")
set(SM_OPTIONS "sm_20" "sm_30" "sm_50" "sm_60" "sm_70")
set_property(CACHE SM PROPERTY STRINGS ${SM_OPTIONS})

set(DEVICE_SUB_ARCH ${SM})

set(DEVICE_BACKEND "CUDA" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "DUMMY" "CUDA")

set(REAL_SIZE_IN_BYTES "8" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE_IN_BYTES PROPERTY STRINGS "8" "4")

add_subdirectory(root)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GMock REQUIRED gmock)

add_executable(tests main.cpp reductions.cpp)
target_link_libraries(tests device ${GMock_LIBRARIES})
target_include_directories(tests PRIVATE root ${GMock_INCLUDE_DIRS})
target_compile_definitions(tests PRIVATE REAL_SIZE=${REAL_SIZE_IN_BYTES})

enable_testing()
add_test(NAME all_tests COMMAND tests)
