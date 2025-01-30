# SPDX-FileCopyrightText: 2019-2024 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.20)
project(examples)

include(../cmake/common_options.cmake)

add_subdirectory(root)
add_executable(basic basic.cpp)
target_include_directories(basic PUBLIC root)
target_link_libraries(basic device)
target_compile_definitions(basic PRIVATE REAL_SIZE=${REAL_SIZE_IN_BYTES})

