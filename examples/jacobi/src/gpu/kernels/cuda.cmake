# SPDX-FileCopyrightText: 2021 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

set(SOURCE_FILES subroutinesCUDA.cu)

add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})
set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

target_compile_definitions(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -DREAL_SIZE=${REAL_SIZE_IN_BYTES}
    >)

target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xptxas -v;
    --expt-relaxed-constexpr;
    >)
