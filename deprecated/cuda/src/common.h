#ifndef CUSTOME_BLAS_COMMON_H
#define CUSTOME_BLAS_COMMON_H

#include <string>

#if REAL_SIZE == 8
typedef double real;
#elif REAL_SIZE == 4
typedef float real;
#else
#  error REAL_SIZE not supported.
#endif


// for Nvidia Quadro P4000
#define MAX_BLOCK_SIZE 1024
#define MAX_SHAREAD_MEM_SIZE 49152


// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);

#endif //CUSTOME_BLAS_COMMON_H
