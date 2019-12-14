#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <stdio.h>
#include <type_traits>

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

// in case of a base array and offsets
template <typename T>
__device__
typename std::enable_if<std::is_floating_point<T>::value, T*>::type
find_data(T *data, unsigned *stride, unsigned blockId) {
  return &data[stride[blockId]];
}


// in case of a base array and offsets
template <typename T>
__device__
typename std::enable_if<std::is_floating_point<T>::value, T*>::type
find_data(T *data, unsigned stride, unsigned blockId) {
  return &data[blockId * stride];
}


// in case of an array of pointers, i.e. real**
template <typename T>
__device__
typename std::remove_pointer<T>::type find_data(T data, unsigned stride, unsigned blockId) {
  return &(data[blockId][stride]);
}


// ------------------------------------------------------------------------------
//                          CUDA COPY-ADD-SCALE KERNELS
// ------------------------------------------------------------------------------
template <typename AT, typename BT, typename T, typename D>
__global__ void kernel_cuda_copy_add_scale(const int m, const int n,
                                           const real alpha, const AT *A, const int lda,
                                           const real beta, BT *B, const int ldb,
                                           T offsets_A,
                                           D offsets_B) {

  const real *matrix_A = find_data(A, offsets_A, blockIdx.x);
  real *matrix_B =  find_data(B, offsets_B, blockIdx.x);

  matrix_B[threadIdx.x + threadIdx.y * ldb] = beta * matrix_B[threadIdx.x + threadIdx.y * ldb]
                                              + alpha * matrix_A[threadIdx.x + threadIdx.y * lda];

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
template <typename AT, typename BT, typename T, typename D>
void device_copy_add_scale(const int m, const int n,
                           const real alpha, const AT *A, const int lda,
                           const real beta, BT *B, const int ldb,
                           T offsets_A,
                           D offsets_B,
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
                                                offsets_A,
                                                offsets_B);
    CUDA_CHECK;
}


// ------------------------------------------------------------------------------
//                              CUDA GEMM KERNELS
// ------------------------------------------------------------------------------

// TODO: generalize for execution with multible blocks
// transa == CblasNoTrans) && (transb == CblasNoTrans)
template <typename AT, typename BT, typename CT, typename T, typename D, typename F>
__global__ void kernel_cuda_blas_gemm_NN(const int m, const int n, const int k,
                                         const real alpha, const AT *A, const int lda,
                                         const BT *B, const int ldb,
                                         const real beta, CT *C, const int ldc,
                                         T offsets_A = 0,
                                         D offsets_B = 0,
                                         F offsets_C = 0)
{

    const real *matrix_A = find_data(A, offsets_A, blockIdx.x);
    const real *matrix_B = find_data(B, offsets_B, blockIdx.x);
    real *matrix_C = find_data(C, offsets_C, blockIdx.x);

    // declare a pointer to array matrix_A within shared memery
    extern __shared__ real sm_A[];

    // declare a pointer to array matrix_B within shared memery
    // NOTE: sm_B and sm_A points into the same chunch of memory
    //       However, sm_B is shifted from sm_A into m * k elements
    real *sm_B = sm_A + m * k;

    // load matrix matrix_A to shared memory
    // slide along matrix columns
    const unsigned column_repeats = (k + n - 1) / n;
    for (int column_shift = 0; column_shift < column_repeats; ++column_shift) {

        int column = threadIdx.y + column_shift * blockDim.y;
        if (column < k)
            sm_A[threadIdx.x + column * m] = matrix_A[threadIdx.x + column * lda];
    }

    // load matrix matrix_B to shared memory
    // slide along matrix rows
    int row_repeats = (k + m - 1) / m;
    for (int row_shift = 0; row_shift < row_repeats; ++row_shift) {
        int row = threadIdx.x + row_shift * blockDim.x;
        if (row < k)
            sm_B[row + threadIdx.y * k] = matrix_B[row + threadIdx.y * ldb];
    }

    __syncthreads();

    real result = 0.0;
    for (int j = 0; j < k; ++j) {
        result += (sm_A[threadIdx.x + j * m] * sm_B[j + threadIdx.y * k]);
    }

  matrix_C[threadIdx.x + threadIdx.y * ldc] = alpha * result
                                              + beta * matrix_C[threadIdx.x + threadIdx.y * ldc];
}


// TODO: generalize for execution with multible blocks
// (transa == CblasTrans) && (transb == CblasTrans)
template <typename AT, typename BT, typename CT, typename T, typename D, typename F>
__global__ void kernel_cuda_blas_gemm_TT(const int m, const int n, const int k,
                                         const real alpha, const AT *A, const int lda,
                                         const BT *B, const int ldb,
                                         const real beta, CT *C, const int ldc,
                                         T offsets_A = 0,
                                         D offsets_B = 0,
                                         F offsets_C = 0)
{

  const real *matrix_A = find_data(A, offsets_A, blockIdx.x);
  const real *matrix_B = find_data(B, offsets_B, blockIdx.x);
  real *matrix_C = find_data(C, offsets_C, blockIdx.x);

    // declare a pointer to array matrix_A within shared memery
    extern __shared__ real sm_A[];

    // declare a pointer to array matrix_B within shared memery
    // NOTE: sm_B and sm_A points into the same chunch of memory
    //       However, sm_B is shifted from sm_A into m * k elements
    real *sm_B = sm_A + m * k;

    // load matrix matrix_A to shared memory
    // slide along matrix columns
    int column_repeats = (m + n - 1) / n;
    for (int column_shift = 0; column_shift < column_repeats; ++column_shift) {
        int column = threadIdx.y + column_shift * blockDim.y;

        // slide along matrix rows
        int row_repeats = (k + m - 1) / m;
        for (int row_shift = 0; row_shift < row_repeats; ++row_shift) {
            int row = threadIdx.x + row_shift * blockDim.x;
            if ((row < k) && (column < m)) {
                sm_A[row + column * k] = matrix_A[row + column * lda];
            }
        }
    }


    // load matrix matrix_B to shared memory
    // slide along matrix columns
    column_repeats = (k + n - 1) / n;
    for (int column_shift = 0; column_shift < column_repeats; ++column_shift) {
        int column = threadIdx.y + column_shift * blockDim.y;

        // slide along matrix rows
        int row_repeats = (n + m - 1) / m;
        for (int row_shift = 0; row_shift < row_repeats; ++row_shift) {
            int row = threadIdx.x + row_shift * blockDim.x;
            if ((row < n) && (column < k)) {
                sm_B[row + column * n] = matrix_B[row + column * ldb];
            }
        }
    }

    __syncthreads();

    real result = 0.0;
    for (int j = 0; j < k; ++j) {
        result += (sm_A[j + threadIdx.x * k] * sm_B[threadIdx.y + j * n]);
    }


  matrix_C[threadIdx.x + threadIdx.y * ldc] = alpha * result
                                              + beta * matrix_C[threadIdx.x + threadIdx.y * ldc];
}


// TODO: generalize for execution with multible blocks
// (transa == CblasTrans) && (transb == CblasNoTrans)
template <typename AT, typename BT, typename CT, typename T, typename D, typename F>
__global__ void kernel_cuda_blas_gemm_TN(const int m, const int n, const int k,
                                         const real alpha, const AT *A, const int lda,
                                         const BT *B, const int ldb,
                                         const real beta, CT *C, const int ldc,
                                         T offsets_A = 0,
                                         D offsets_B = 0,
                                         F offsets_C = 0)
{

  const real *matrix_A = find_data(A, offsets_A, blockIdx.x);
  const real *matrix_B = find_data(B, offsets_B, blockIdx.x);
  real *matrix_C = find_data(C, offsets_C, blockIdx.x);

    // declare a pointer to array matrix_A within shared memery
    extern __shared__ real sm_A[];

    // declare a pointer to array matrix_B within shared memery
    // NOTE: sm_B and sm_A points into the same chunch of memory
    //       However, sm_B is shifted from sm_A into m * k elements
    real *sm_B = sm_A + m * k;

    // load matrix matrix_A to shared memory
    // slide along matrix columns
    int column_repeats = (m + n - 1) / n;
    for (int column_shift = 0; column_shift < column_repeats; ++column_shift) {
        int column = threadIdx.y + column_shift * blockDim.y;

        // slide along matrix rows
        int row_repeats = (k + m - 1) / m;
        for (int row_shift = 0; row_shift < row_repeats; ++row_shift) {
            int row = threadIdx.x + row_shift * blockDim.x;
            if ((row < k) && (column < m)) {
                sm_A[row + column * k] = matrix_A[row + column * lda];
            }
        }
    }

    // load matrix matrix_B to shared memory
    // slide along matrix rows
    int row_repeats = (k + m - 1) / m;
    for (int row_shift = 0; row_shift < row_repeats; ++row_shift) {
        int row = threadIdx.x + row_shift * blockDim.x;
        if (row < k)
            sm_B[row + threadIdx.y * k] = matrix_B[row + threadIdx.y * ldb];
    }

    __syncthreads();

    real result = 0.0;
    for (int j = 0; j < k; ++j) {
        result += (sm_A[j + threadIdx.x * k] * sm_B[j + threadIdx.y * k]);
    }


  matrix_C[threadIdx.x + threadIdx.y * ldc] = alpha * result
                                              + beta * matrix_C[threadIdx.x + threadIdx.y * ldc];
}



// TODO: generalize for execution with multible blocks
// (transa == CblasNoTrans) && (transb == CblasTrans)
template <typename AT, typename BT, typename CT, typename T, typename D, typename F>
__global__ void kernel_cuda_blas_gemm_NT(const int m, const int n, const int k,
                                         const real alpha, const AT *A, const int lda,
                                         const BT *B, const int ldb,
                                         const real beta, CT *C, const int ldc,
                                         T offsets_A = 0,
                                         D offsets_B = 0,
                                         F offsets_C = 0)
{

  const real *matrix_A = find_data(A, offsets_A, blockIdx.x);
  const real *matrix_B = find_data(B, offsets_B, blockIdx.x);
  real *matrix_C = find_data(C, offsets_C, blockIdx.x);

    // declare a pointer to array matrix_A within shared memery
    extern __shared__ real sm_A[];

    // declare a pointer to array matrix_B within shared memery
    // NOTE: sm_B and sm_A points into the same chunch of memory
    //       However, sm_B is shifted from sm_A into m * k elements
    real *sm_B = sm_A + m * k;

   // load matrix matrix_A to shared memory
    // slide along matrix columns
    int column_repeats = (k + n - 1) / n;
    for (int column_shift = 0; column_shift < column_repeats; ++column_shift) {
        int column = threadIdx.y + column_shift * blockDim.y;
        if (column < k)
            sm_A[threadIdx.x + column * m] = matrix_A[threadIdx.x + column * lda];
    }


    // load matrix matrix_B to shared memory
    // slide along matrix columns
    column_repeats = (k + n - 1) / n;
    for (int column_shift = 0; column_shift < column_repeats; ++column_shift) {
        int column = threadIdx.y + column_shift * blockDim.y;

        // slide along matrix rows
        int row_repeats = (n + m - 1) / m;
        for (int row_shift = 0; row_shift < row_repeats; ++row_shift) {
            int row = threadIdx.x + row_shift * blockDim.x;
            if ((row < n) && (column < k)) {
                sm_B[row + column * n] = matrix_B[row + column * ldb];
            }
        }
    }

    __syncthreads();

    real result = 0.0;
    for (int j = 0; j < k; ++j) {
        result += (sm_A[threadIdx.x + j * m] * sm_B[threadIdx.y + j * n]);
    }


  matrix_C[threadIdx.x + threadIdx.y * ldc] = alpha * result
                                              + beta * matrix_C[threadIdx.x + threadIdx.y * ldc];
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
template <typename AT, typename BT, typename CT, typename T, typename D, typename F>
void device_gemm(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE transa,
                 const CBLAS_TRANSPOSE transb,
                 const int m, const int n, const int k,
                 const real alpha, const AT *A_base, const int lda,
                 const BT *B_base, const int ldb,
                 const real beta, CT *C_base, const int ldc,
                 T offsets_A,
                 D offsets_B,
                 F offsets_C,
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
                                                                       offsets_A,
                                                                       offsets_B,
                                                                       offsets_C);
            CUDA_CHECK;
        }
        else if ((transa == CblasTrans) && (transb == CblasTrans)) {
            kernel_cuda_blas_gemm_TT<<<grid, block, shared_mem_size>>>(m, n, k,
                                                                       alpha, A_base, lda,
                                                                       B_base, ldb,
                                                                       beta, C_base, ldc,
                                                                       offsets_A,
                                                                       offsets_B,
                                                                       offsets_C);
            CUDA_CHECK;
        }
        else if ((transa == CblasNoTrans) && (transb == CblasTrans)) {
            kernel_cuda_blas_gemm_NT<<<grid, block, shared_mem_size>>>(m, n, k,
                                                                       alpha, A_base, lda,
                                                                       B_base, ldb,
                                                                       beta, C_base, ldc,
                                                                       offsets_A,
                                                                       offsets_B,
                                                                       offsets_C);
            CUDA_CHECK;
        }
        else if ((transa == CblasTrans) && (transb == CblasNoTrans)) {
            kernel_cuda_blas_gemm_TN<<<grid, block, shared_mem_size>>>(m, n, k,
                                                                       alpha, A_base, lda,
                                                                       B_base, ldb,
                                                                       beta, C_base, ldc,
                                                                       offsets_A,
                                                                       offsets_B,
                                                                       offsets_C);
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
* NOTE: we use pragma noinline to prevent inlining during a code optimization i.e. -O2.
*   if compiler inlines a function then the corresponding template specification is not going to appear
*   as a symbol in a static library
*/



#include <tuple>
#include <array>
#include <type_traits>

void instantiate() {
    unsigned int integer = 0;
    unsigned int *uint_ptr = 0;
    real *data = 0;

    //------------------------------------------------------------------------------------------------------------------
    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, integer, integer, integer, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, uint_ptr, integer, integer, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, integer, uint_ptr, integer, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, integer, integer, uint_ptr, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, uint_ptr, uint_ptr, integer, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, uint_ptr, integer, uint_ptr, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, integer, uint_ptr, uint_ptr, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data, 0, uint_ptr, uint_ptr, uint_ptr, 0);


    //------------------------------------------------------------------------------------------------------------------
    real **data_array = nullptr;

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data_array, 0, data, 0, 0.0, data, 0, integer, integer, integer, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data_array, 0, 0.0, data, 0, integer, integer, integer, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data, 0, 0.0, data_array, 0, integer, integer, integer, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data_array, 0, data_array, 0, 0.0, data, 0, integer, integer, integer, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data_array, 0, data, 0, 0.0, data_array, 0, integer, integer, integer, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data, 0, data_array, 0, 0.0, data_array, 0, integer, integer, integer, 0);

    #pragma noinline
    device_gemm(CblasColMajor, CblasTrans, CblasTrans, 0, 0, 0, 0.0, data_array, 0, data_array, 0, 0.0, data_array, 0, integer, integer, integer, 0);


#pragma noinline
  device_copy_add_scale(0, 0, 0.0, data, 0, 0.0, data, 0, integer, integer, 0);

  #pragma noinline
  device_copy_add_scale(0, 0, 0.0, data, 0, 0.0, data, 0, uint_ptr, integer, 0);

  #pragma noinline
  device_copy_add_scale(0, 0, 0.0, data, 0, 0.0, data, 0, integer, uint_ptr, 0);

  #pragma noinline
  device_copy_add_scale(0, 0, 0.0, data, 0, 0.0, data, 0, uint_ptr, uint_ptr, 0);

  //--------------------------------------------------------------------------------------------------------------------

#pragma noinline
  device_copy_add_scale(0, 0, 0.0, data, 0, 0.0, data, 0, integer, integer, 0);

#pragma noinline
  device_copy_add_scale(0, 0, 0.0, data_array, 0, 0.0, data, 0, integer, integer, 0);

#pragma noinline
  device_copy_add_scale(0, 0, 0.0, data, 0, 0.0, data_array, 0, integer, integer, 0);

#pragma noinline
  device_copy_add_scale(0, 0, 0.0, data_array, 0, 0.0, data_array, 0, integer, integer, 0);


}