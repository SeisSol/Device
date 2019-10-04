#ifndef CUDA_UTILS_CUH_
#define CUDA_UTILS_CUH_

#include <string>
#include "common.h"


// memory menagment:
void check_device_operation();
void * device_malloc(size_t size);
void * device_malloc_pinned(size_t size);

void device_free(void *devPtr);
void device_free_pinned(void *devPtr);
void device_synch();


void device_copy_to(void* dst, const void* src, size_t count);
void device_copy_from(void* dst, const void* src, size_t count);
void device_copy_between(void* dst, const void* src, size_t count);

void device_copy_2D_to(void *dst,
                       size_t dpitch,
                       const void *src,
                       size_t spitch,
                       size_t width,
                       size_t height);

void device_copy_2D_from(void *dst,
                         size_t dpitch,
                         const void *src,
                         size_t spitch,
                         size_t width,
                         size_t height);


void device_compare_with_host_array(const real *host_ptr,
                                    const real * dev_ptr,
                                    const size_t num_elements,
                                    const char *array_name = NULL);

void device_init_linspace(const size_t num_elements, unsigned *array);
void device_scale_linspace(const unsigned* linspace, const unsigned scale, const size_t num_elements, unsigned *output);
void device_scale_array(const real scalar, const size_t num_elements, real *dev_array);
void device_scale_array(const unsigned scale, const size_t num_elements, unsigned *output);
void device_init_array(const unsigned value, const size_t num_elements, unsigned *output);


// computational kernels:
#if !defined CBLAS_H && !defined __MKL_CBLAS_H__
    enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102};
    enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
#endif


template <typename T, typename D>
void cuda_copy_add_scale(const int m, const int n,
                         const real alpha, const real *A, const int lda,
                         const real beta, real *B, const int ldb,
                         T stride_A,
                         D stride_B,
                         const unsigned num_elements);


template <typename T, typename D, typename F>
void cuda_blas_gemm(const CBLAS_LAYOUT Layout,
                    const CBLAS_TRANSPOSE transa,
                    const CBLAS_TRANSPOSE transb,
                    const int m, const int n, const int k,
                    const real alpha, const real *A_base, const int lda,
                    const real *B_base, const int ldb,
                    const real beta, real *C_base, const int ldc,
                    T stride_A,
                    D stride_B,
                    F stride_C,
                    const unsigned num_elements = 1);

#endif  // CUDA_UTILS_CUH_