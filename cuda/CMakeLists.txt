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
    set(HARDWARE_DEFINITIONS "REAL_SIZE=4" "ALIGNMENT=128")
endif()
add_cuda_defenitions("${HARDWARE_DEFINITIONS}")  # returns:  AUX_FLAGS


#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3; -std=c++11; --use_fast_math; -arch=sm_61; -Xptxas -v; ${AUX_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11; -arch=sm_50; -Xptxas -v; ${AUX_FLAGS})

cuda_add_library(custom_blas STATIC "src/auxiliary.cu"
                                    "src/gemm.cu"
                                    "src/memory_menagment.cu"
                                    "src/scratch_mem_system.cu")

target_link_libraries(custom_blas -lnvToolsExt)