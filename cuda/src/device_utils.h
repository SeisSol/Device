
#ifndef CUDA_UTILS_CUH_
#define CUDA_UTILS_CUH_

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
#define CUDA_CHECK device_check(__FILE__,__LINE__)
void device_check(std::string file, int line);


void check_device_operation();


#if !defined CBLAS_H && !defined __MKL_CBLAS_H__
    enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102};
    enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
#endif


void * device_malloc(size_t size);
void device_copy_to(void* dst, const void* src, size_t count);
void device_copy_from(void* dst, const void* src, size_t count);
void device_free(void *devPtr);


void device_scalar_tensor_mult(const real scalar, 
                             const real *lhs_tensor,
                             real *rhs_tensor,
                             const unsigned tensor_size,
                             const unsigned num_element = 1);



void device_scalar_tensor_mult_add(const real scalar, 
                                 const real *lhs_tensor,
                                 real *rhs_tensor,
                                 const unsigned tensor_size,
                                 const unsigned num_element = 1);


void device_copy_add_scale(const int m, const int n, 
                         const real alpha, const real *A, const int lda, 
                         const real beta, real *B, const int ldb,
                         const unsigned jump_to_next_tensor_A,
                         const unsigned jump_to_next_tensor_B,
                         const unsigned num_elements);


void device_blas_gemm(const CBLAS_LAYOUT Layout, 
                    const CBLAS_TRANSPOSE transa, 
                    const CBLAS_TRANSPOSE transb, 
                    const int m, const int n, const int k, 
                    const real alpha, const real *A, const int lda, 
                    const real *B, const int ldb, 
                    const real beta, real *C, const int ldc,
                    const unsigned jump_to_next_tensor_A = 0,
                    const unsigned jump_to_next_tensor_B = 0,
                    const unsigned jump_to_next_tensor_C = 0,
                    const unsigned num_elements = 1);

#endif  // CUDA_UTILS_CUH_