cmake_minimum_required(VERSION 3.5)
project(cuda_custom_blas)

function(add_cuda_defenitions definitions)
    foreach(item ${definitions})
        list(APPEND augmented_item_list "-D${item}")
    endforeach(item)

    set(AUX_FLAGS ${augmented_item_list} PARENT_SCOPE)
endfunction()


find_package(CUDA REQUIRED)


if (NOT DEFINED HARDWARE_DEFINITIONS)
    set(FATAL_ERROR "HARDWARE_DEFINITIONS variable is not set. \
        It is a list that must contain the following: REAL_SIZE, ALIGNMENT")
endif()
message("HARDWARE_DEFINITIONS: ${HARDWARE_DEFINITIONS}")
add_cuda_defenitions("${HARDWARE_DEFINITIONS}")  # returns:  AUX_FLAGS

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3; -std=c++11; --use_fast_math; -arch=sm_61; -Xptxas -v; ${AUX_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11; -arch=sm_50; -Xptxas -v; ${AUX_FLAGS})

cuda_add_library(custom_blas STATIC "src/auxiliary.cu"
                                    "src/gemm.cu"
                                    "src/memory_menagment.cu"
                                    "src/temporary_mem_menager.cu"
                                    "src/device_query.cu")

target_link_libraries(custom_blas -lnvToolsExt -lcuda)
