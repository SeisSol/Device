set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Compile the CUDA kernel
find_package(CUDA REQUIRED)

#Set the source files and needed flags
set(SOURCE_FILES subroutinesCUDA.cu)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                    -DREAL_SIZE=${REAL_SIZE_IN_BYTES};
                    --compiler-options -fPIC;
                    -std=c++11)

#Compile
cuda_add_library(${TARGET_NAME} ${SOURCE_FILES})