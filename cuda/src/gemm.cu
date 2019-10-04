#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <stdio.h>

#include "common.h"


#if !defined CBLAS_H && !defined __MKL_CBLAS_H__
    enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102};
    enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
#endif


extern dim3 compute_grid_1D(const dim3 &block, const int size);


__device__ unsigned get_stride(unsigned *stride, unsigned blockId) {
    return stride[blockId];
}

__device__ unsigned get_stride(unsigned stride, unsigned blockId) {
    return blockId * stride;
}


// ------------------------------------------------------------------------------
//                          CUDA COPY-ADD-SCALE KERNELS
// ------------------------------------------------------------------------------
template <typename T, typename D>
__global__ void kernel_cuda_copy_add_scale(const int m, const int n,
                                           const real alpha, const real *A, const int lda,
                                           const real beta, real *B, const int ldb,
                                           T stride_A,
                                           D stride_B) {

    A += get_stride(stride_A, blockIdx.x);
    B += get_stride(stride_B, blockIdx.x);

    B[threadIdx.x + threadIdx.y * ldb] = beta  * B[threadIdx.x + threadIdx.y * ldb]
                                       + alpha * A[threadIdx.x + threadIdx.y * lda];

}

/**
 *  Implementation of basic GEMM operation
 *  on GPU using CUDA
 *
 *  C = alpha * A * B + beta * C
 *
 *    B                    B                   A
 *
 *   _n____              _n____           _n_______
 *  |   .  |            |   .  |         |   .     |
 *  |m  .  | = brta *   |m  .  | + alpha |m  .     |
 *  |....  |            |....  |         |....     | lda
 *  |      |            |      |         |_________|
 *  |      |            |      | ldb
 *  |______|            |______|
 *
 * */
 template <typename T, typename D>
void cuda_copy_add_scale(const int m, const int n,
                         const real alpha, const real *A, const int lda,
                         const real beta, real *B, const int ldb,
                         T stride_A,
                         D stride_B,
                         const unsigned num_elements)
{

    dim3 block(m, n, 1); // block(x, y, z)
    dim3 grid (num_elements, 1, 1); // grid(x, y, z)


    assert((m * n) <= MAX_BLOCK_SIZE);

    assert((m <= lda) && (m <= ldb));
    assert((beta == 1.0) || (beta == 0.0) || (beta == -1.0));


    kernel_cuda_copy_add_scale<<<grid, block>>>(m, n,
                                                alpha, A, lda,
                                                beta, B, ldb,
                                                stride_A,
                                                stride_B);
    CUDA_CHECK;
}


// ------------------------------------------------------------------------------
//                              CUDA GEMM KERNELS
// ------------------------------------------------------------------------------

// TODO: generalize for execution with multible blocks
// transa == CblasNoTrans) && (transb == CblasNoTrans)
template <typename T, typename D, typename F>
__global__ void kernel_cuda_blas_gemm_NN(const int m, const int n, const int k,
                                         const real alpha, const real *A, const int lda,
                                         const real *B, const int ldb,
                                         const real beta, real *C, const int ldc,
                                         T stride_A = 0,
                                         D stride_B = 0,
                                         F stride_C = 0)
{

    A += get_stride(stride_A, blockIdx.x);
    B += get_stride(stride_B, blockIdx.x);
    C += get_stride(stride_C, blockIdx.x);

    // declare a pointer to array A within shared memery
    extern __shared__ real sm_A[];

    // declare a pointer to array B within shared memery
    // NOTE: sm_B and sm_A points into the same chunch of memory
    //       However, sm_B is shifted from sm_A into m * k elements
    real *sm_B = sm_A + m * k;

    // load matrix A to shared memory
    // slide along matrix columns
    const unsigned column_repeats = (k + n - 1) / n;
    for (int column_shift = 0; column_shift < column_repeats; ++column_shift) {

        int column = threadIdx.y + column_shift * blockDim.y;
        if (column < k)
            sm_A[threadIdx.x + column * m] = A[threadIdx.x + column * lda];
    }

    // load matrix B to shared memory
    // slide along matrix rows
    int row_repeats = (k + m - 1) / m;
    for (int row_shift = 0; row_shift < row_repeats; ++row_shift) {
        int row = threadIdx.x + row_shift * blockDim.x;
        if (row < k)
            sm_B[row + threadIdx.y * k] = B[row + threadIdx.y * ldb];
    }

    __syncthreads();

    real result = 0.0;
    for (int j = 0; j < k; ++j) {
        result += (sm_A[threadIdx.x + j * m] * sm_B[j + threadIdx.y * k]);
    }

    C[threadIdx.x + threadIdx.y * ldc] = alpha * result
                                       + beta * C[threadIdx.x + threadIdx.y * ldc];
}


// TODO: generalize for execution with multible blocks
// (transa == CblasTrans) && (transb == CblasTrans)
template <typename T, typename D, typename F>
__global__ void kernel_cuda_blas_gemm_TT(const int m, const int n, const int k,
                                         const real alpha, const real *A, const int lda,
                                         const real *B, const int ldb,
                                         const real beta, real *C, const int ldc,
                                         T stride_A = 0,
                                         D stride_B = 0,
                                         F stride_C = 0)
{

    A += get_stride(stride_A, blockIdx.x);
    B += get_stride(stride_B, blockIdx.x);
    C += get_stride(stride_C, blockIdx.x);

    // declare a pointer to array A within shared memery
    extern __shared__ real sm_A[];

    // declare a pointer to array B within shared memery
    // NOTE: sm_B and sm_A points into the same chunch of memory
    //       However, sm_B is shifted from sm_A into m * k elements
    real *sm_B = sm_A + m * k;

    // load matrix A to shared memory
    // slide along matrix columns
    int column_repeats = (m + n - 1) / n;
    for (int column_shift = 0; column_shift < column_repeats; ++column_shift) {
        int column = threadIdx.y + column_shift * blockDim.y;

        // slide along matrix rows
        int row_repeats = (k + m - 1) / m;
        for (int row_shift = 0; row_shift < row_repeats; ++row_shift) {
            int row = threadIdx.x + row_shift * blockDim.x;
            if ((row < k) && (column < m)) {
                sm_A[row + column * k] = A[row + column * lda];
            }
        }
    }


    // load matrix B to shared memory
    // slide along matrix columns
    column_repeats = (k + n - 1) / n;
    for (int column_shift = 0; column_shift < column_repeats; ++column_shift) {
        int column = threadIdx.y + column_shift * blockDim.y;

        // slide along matrix rows
        int row_repeats = (n + m - 1) / m;
        for (int row_shift = 0; row_shift < row_repeats; ++row_shift) {
            int row = threadIdx.x + row_shift * blockDim.x;
            if ((row < n) && (column < k)) {
                sm_B[row + column * n] = B[row + column * ldb];
            }
        }
    }

    __syncthreads();

    real result = 0.0;
    for (int j = 0; j < k; ++j) {
        result += (sm_A[j + threadIdx.x * k] * sm_B[threadIdx.y + j * n]);
    }


    C[threadIdx.x + threadIdx.y * ldc] = alpha * result
                                       + beta * C[threadIdx.x + threadIdx.y * ldc];
}


// TODO: generalize for execution with multible blocks
// (transa == CblasTrans) && (transb == CblasNoTrans)
template <typename T, typename D, typename F>
__global__ void kernel_cuda_blas_gemm_TN(const int m, const int n, const int k,
                                         const real alpha, const real *A, const int lda,
                                         const real *B, const int ldb,
                                         const real beta, real *C, const int ldc,
                                         T stride_A = 0,
                                         D stride_B = 0,
                                         F stride_C = 0)
{

    A += get_stride(stride_A, blockIdx.x);
    B += get_stride(stride_B, blockIdx.x);
    C += get_stride(stride_C, blockIdx.x);

    // declare a pointer to array A within shared memery
    extern __shared__ real sm_A[];

    // declare a pointer to array B within shared memery
    // NOTE: sm_B and sm_A points into the same chunch of memory
    //       However, sm_B is shifted from sm_A into m * k elements
    real *sm_B = sm_A + m * k;

    // load matrix A to shared memory
    // slide along matrix columns
    int column_repeats = (m + n - 1) / n;
    for (int column_shift = 0; column_shift < column_repeats; ++column_shift) {
        int column = threadIdx.y + column_shift * blockDim.y;

        // slide along matrix rows
        int row_repeats = (k + m - 1) / m;
        for (int row_shift = 0; row_shift < row_repeats; ++row_shift) {
            int row = threadIdx.x + row_shift * blockDim.x;
            if ((row < k) && (column < m)) {
                sm_A[row + column * k] = A[row + column * lda];
            }
        }
    }

    // load matrix B to shared memory
    // slide along matrix rows
    int row_repeats = (k + m - 1) / m;
    for (int row_shift = 0; row_shift < row_repeats; ++row_shift) {
        int row = threadIdx.x + row_shift * blockDim.x;
        if (row < k)
            sm_B[row + threadIdx.y * k] = B[row + threadIdx.y * ldb];
    }

    __syncthreads();

    real result = 0.0;
    for (int j = 0; j < k; ++j) {
        result += (sm_A[j + threadIdx.x * k] * sm_B[j + threadIdx.y * k]);
    }


    C[threadIdx.x + threadIdx.y * ldc] = alpha * result
                                       + beta * C[threadIdx.x + threadIdx.y * ldc];
}



// TODO: generalize for execution with multible blocks
// (transa == CblasNoTrans) && (transb == CblasTrans)
template <typename T, typename D, typename F>
__global__ void kernel_cuda_blas_gemm_NT(const int m, const int n, const int k,
                                         const real alpha, const real *A, const int lda,
                                         const real *B, const int ldb,
                                         const real beta, real *C, const int ldc,
                                         T stride_A = 0,
                                         D stride_B = 0,
                                         F stride_C = 0)
{
    A += get_stride(stride_A, blockIdx.x);
    B += get_stride(stride_B, blockIdx.x);
    C += get_stride(stride_C, blockIdx.x);

    // declare a pointer to array A within shared memery
    extern __shared__ real sm_A[];

    // declare a pointer to array B within shared memery
    // NOTE: sm_B and sm_A points into the same chunch of memory
    //       However, sm_B is shifted from sm_A into m * k elements
    real *sm_B = sm_A + m * k;

   // load matrix A to shared memory
    // slide along matrix columns
    int column_repeats = (k + n - 1) / n;
    for (int column_shift = 0; column_shift < column_repeats; ++column_shift) {
        int column = threadIdx.y + column_shift * blockDim.y;
        if (column < k)
            sm_A[threadIdx.x + column * m] = A[threadIdx.x + column * lda];
    }


    // load matrix B to shared memory
    // slide along matrix columns
    column_repeats = (k + n - 1) / n;
    for (int column_shift = 0; column_shift < column_repeats; ++column_shift) {
        int column = threadIdx.y + column_shift * blockDim.y;

        // slide along matrix rows
        int row_repeats = (n + m - 1) / m;
        for (int row_shift = 0; row_shift < row_repeats; ++row_shift) {
            int row = threadIdx.x + row_shift * blockDim.x;
            if ((row < n) && (column < k)) {
                sm_B[row + column * n] = B[row + column * ldb];
            }
        }
    }

    __syncthreads();

    real result = 0.0;
    for (int j = 0; j < k; ++j) {
        result += (sm_A[threadIdx.x + j * m] * sm_B[threadIdx.y + j * n]);
    }


    C[threadIdx.x + threadIdx.y * ldc] = alpha * result
                                       + beta * C[threadIdx.x + threadIdx.y * ldc];
}



// ------------------------------------------------------------------------------
// NOTE: it works only with a grid size equal to one

/**
 *  Implementation of basic GEMM operation
 *  on GPU using CUDA
 *
 *  C = alpha * A * B + beta * C
 *
 *    C                A        B              C
 *
 *   _n_             __k__     _n_            _n_
 *  |   |           |     |   |   |          |   |
 *  | C | = alpha * |m    | x |k  | + beta * |   |
 *  |   |           |     |   |   |          |   |
 *  |m  |           |     |   |___|          |m  |
 *  |   |           |     |                  |   |
 *  |___|           |_____|                  |___|
 *
 * */
#include <iostream>  // DEBUGGING
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
                    const unsigned num_elements) {
    dim3 block(m, n, 1); // block(x, y, z)
    dim3 grid (num_elements, 1, 1); // grid(x, y, z)
    size_t shared_mem_size = (m * k + k * n) * sizeof(real);

    assert((m * n) <= MAX_BLOCK_SIZE);
    assert(shared_mem_size <= MAX_SHAREAD_MEM_SIZE);

    if (Layout == CblasColMajor) {


        // perform initial check: similar to netlib blas implementation
        assert((m != 0) && (n != 0) && (k != 0));
        assert(ldc >= std::max(1, m));

        const int num_rows_A = transa == CblasNoTrans ? m : k;
        const int num_rows_B = transb == CblasNoTrans ? k : n;
        assert(lda >= std::max(1, num_rows_A));
        assert(ldb >= std::max(1, num_rows_B));

        if ((transa == CblasNoTrans) && (transb == CblasNoTrans)) {
            kernel_cuda_blas_gemm_NN<<<grid, block, shared_mem_size>>>(m, n, k,
                                                                       alpha, A_base, lda,
                                                                       B_base, ldb,
                                                                       beta, C_base, ldc,
                                                                       stride_A,
                                                                       stride_B,
                                                                       stride_C);
            CUDA_CHECK;
        }
        else if ((transa == CblasTrans) && (transb == CblasTrans)) {
            kernel_cuda_blas_gemm_TT<<<grid, block, shared_mem_size>>>(m, n, k,
                                                                       alpha, A_base, lda,
                                                                       B_base, ldb,
                                                                       beta, C_base, ldc,
                                                                       stride_A,
                                                                       stride_B,
                                                                       stride_C);
            CUDA_CHECK;
        }
        else if ((transa == CblasNoTrans) && (transb == CblasTrans)) {
            kernel_cuda_blas_gemm_NT<<<grid, block, shared_mem_size>>>(m, n, k,
                                                                       alpha, A_base, lda,
                                                                       B_base, ldb,
                                                                       beta, C_base, ldc,
                                                                       stride_A,
                                                                       stride_B,
                                                                       stride_C);
            CUDA_CHECK;
        }
        else if ((transa == CblasTrans) && (transb == CblasNoTrans)) {
            kernel_cuda_blas_gemm_TN<<<grid, block, shared_mem_size>>>(m, n, k,
                                                                       alpha, A_base, lda,
                                                                       B_base, ldb,
                                                                       beta, C_base, ldc,
                                                                       stride_A,
                                                                       stride_B,
                                                                       stride_C);
            CUDA_CHECK;
        }
        else if ((transa == CblasConjTrans) || (transb == CblasConjTrans)) {
            std::cerr << "ERROR: the library doesn't support matrix conjugate matrix operations" << std::endl;
            exit(-1);
        }
    }
    else {
        std::cerr << "ERROR: the library doesn't row major gemm operation" << std::endl;
        exit(-1);
    }
}








/** The function is used by the compiler to instantiate specific GEMM implementations
* NOTE: it is not callable
* NOTE: we use pragma noinline to prevent inlining during code optimization i.e. -O2.
*   if compiler inlines a function then the corresponding template specification is not going to appear
*   as a symbol in a static library
*/

void instantiate() {
    unsigned int integer = 0;
    unsigned int *ptr = 0;
    real *data = 0;

    #pragma noinline
    cuda_blas_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, integer, integer, integer, 0);

    #pragma noinline
    cuda_blas_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, ptr, integer, integer, 0);

    #pragma noinline
    cuda_blas_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, integer, ptr, integer, 0);

    #pragma noinline
    cuda_blas_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, integer, integer, ptr, 0);

    #pragma noinline
    cuda_blas_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, ptr, ptr, integer, 0);

    #pragma noinline
    cuda_blas_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, ptr, integer, ptr, 0);

    #pragma noinline
    cuda_blas_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, integer, ptr, ptr, 0);

    #pragma noinline
    cuda_blas_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, ptr, ptr, ptr, 0);

    #pragma noinline
    cuda_copy_add_scale<unsigned int, unsigned int>(0, 0, 0.0, data, 0, 0.0, data, 0, integer, integer, 0);

    #pragma noinline
    cuda_copy_add_scale<unsigned int*, unsigned int>(0, 0, 0.0, data, 0, 0.0, data, 0, ptr, integer, 0);

    #pragma noinline
    cuda_copy_add_scale<unsigned int, unsigned int*>(0, 0, 0.0, data, 0, 0.0, data, 0, integer, ptr, 0);

    #pragma noinline
    cuda_copy_add_scale<unsigned int*, unsigned int*>(0, 0, 0.0, data, 0, 0.0, data, 0, ptr, ptr, 0);
}