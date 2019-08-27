#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <assert.h>
#include "device_utils.h"
#include <stdio.h>


#ifndef CUDA_CHECK
    #define CUDA_CHECK cuda_check(__FILE__,__LINE__)
#endif


std::string prev_file = "";
int prev_line = 0;
void device_check(std::string file, int line) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << std::endl << file << ", line " << line 
                  << ": " << cudaGetErrorString(e) << " (" << e << ")" << std::endl;
        if (prev_line > 0)
            std::cout << "Previous CUDA call:" << std::endl 
                      << prev_file << ", line " << prev_line << std::endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}


inline dim3 compute_grid_1D(const dim3 &block, const int size) {
    int num_blocks_x = (size + block.x - 1) / block.x;
    return dim3(num_blocks_x, 1, 1);
}


// ------------------------------------------------------------------------------
__global__ void kernel_gpu_operation_check() {
    printf("gpu offloading is working\n");
}

void check_device_operation() {
    std::cout << "checking gpu offloading" << std::endl;
    kernel_gpu_operation_check<<<1,1>>>(); CUDA_CHECK;
    cudaDeviceSynchronize();
}

// ------------------------------------------------------------------------------
__global__ void kernel_cuda_scalar_tensor_mult(const real scalar, 
                                               const real *lhs_tensor,
                                               real *rhs_tensor, 
                                               const long long size) {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        rhs_tensor[idx] = scalar * lhs_tensor[idx];
}


void device_scalar_tensor_mult(const real scalar, 
                              const real *lhs_tensor,
                              real *rhs_tensor,
                              const unsigned tensor_size,
                              const unsigned num_element) {
    
    // TODO: adjust grid and block
    long long num_floats = tensor_size * num_element;
    unsigned optimal_kernel_size = 64;
    dim3 block(optimal_kernel_size, 1, 1);
    dim3 grid = compute_grid_1D(block,  num_floats);
    kernel_cuda_scalar_tensor_mult<<<grid, block>>>(scalar, lhs_tensor, rhs_tensor, num_floats); 
    CUDA_CHECK;
}


// ------------------------------------------------------------------------------
__global__ void kernel_cuda_scalar_tensor_mult_add(const real scalar, 
                                                   const real *lhs_tensor,
                                                   real *rhs_tensor, 
                                                   const long long size) {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        rhs_tensor[idx] += scalar * lhs_tensor[idx];
}


void device_scalar_tensor_mult_add(const real scalar, 
                                   const real *lhs_tensor,
                                   real *rhs_tensor,
                                   const unsigned tensor_size,
                                   const unsigned num_element) {
    
    // TODO: adjust grid and block
    long long num_floats = tensor_size * num_element;
    unsigned optimal_kernel_size = 64;
    dim3 block(optimal_kernel_size, 1, 1);
    dim3 grid = compute_grid_1D(block,  num_floats);
    kernel_cuda_scalar_tensor_mult_add<<<grid, block>>>(scalar, lhs_tensor, rhs_tensor, num_floats); 
    CUDA_CHECK;
}
// ------------------------------------------------------------------------------
//                          CUDA COPY-ADD-SCALE KERNELS
// ------------------------------------------------------------------------------
__global__ void kernel_cuda_copy_add_scale(const int m, const int n, 
                                           const real alpha, const real *A, const int lda, 
                                           const real beta, real *B, const int ldb,
                                           const unsigned jump_to_next_tensor_A,
                                           const unsigned jump_to_next_tensor_B) {

    A += blockIdx.x * jump_to_next_tensor_A;
    B += blockIdx.x * jump_to_next_tensor_B;

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
void device_copy_add_scale(const int m, const int n, 
                           const real alpha, const real *A, const int lda, 
                           const real beta, real *B, const int ldb,
                           const unsigned jump_to_next_tensor_A,
                           const unsigned jump_to_next_tensor_B,
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
                                                jump_to_next_tensor_A,
                                                jump_to_next_tensor_B);
    CUDA_CHECK; 
}




// ------------------------------------------------------------------------------
//                              CUDA GEMM KERNELS
// ------------------------------------------------------------------------------

// TODO: generalize for execution with multible blocks
// transa == CblasNoTrans) && (transb == CblasNoTrans)
__global__ void kernel_cuda_blas_gemm_NN(const int m, const int n, const int k,
                                         const real alpha, const real *A, const int lda, 
                                         const real *B, const int ldb,
                                         const real beta, real *C, const int ldc,
                                         const unsigned jump_to_next_tensor_A = 0,
                                         const unsigned jump_to_next_tensor_B = 0,
                                         const unsigned jump_to_next_tensor_C = 0)
{

    A += blockIdx.x * jump_to_next_tensor_A;
    B += blockIdx.x * jump_to_next_tensor_B;
    C += blockIdx.x * jump_to_next_tensor_C;

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
__global__ void kernel_cuda_blas_gemm_TT(const int m, const int n, const int k,
                                         const real alpha, const real *A, const int lda, 
                                         const real *B, const int ldb,
                                         const real beta, real *C, const int ldc,
                                         const unsigned jump_to_next_tensor_A = 0,
                                         const unsigned jump_to_next_tensor_B = 0,
                                         const unsigned jump_to_next_tensor_C = 0)
{

    A += blockIdx.x * jump_to_next_tensor_A;
    B += blockIdx.x * jump_to_next_tensor_B;
    C += blockIdx.x * jump_to_next_tensor_C;

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
__global__ void kernel_cuda_blas_gemm_TN(const int m, const int n, const int k,
                                         const real alpha, const real *A, const int lda, 
                                         const real *B, const int ldb,
                                         const real beta, real *C, const int ldc,
                                         const unsigned jump_to_next_tensor_A = 0,
                                         const unsigned jump_to_next_tensor_B = 0,
                                         const unsigned jump_to_next_tensor_C = 0)
{

    A += blockIdx.x * jump_to_next_tensor_A;
    B += blockIdx.x * jump_to_next_tensor_B;
    C += blockIdx.x * jump_to_next_tensor_C;

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
__global__ void kernel_cuda_blas_gemm_NT(const int m, const int n, const int k,
                                         const real alpha, const real *A, const int lda, 
                                         const real *B, const int ldb,
                                         const real beta, real *C, const int ldc,
                                         const unsigned jump_to_next_tensor_A = 0,
                                         const unsigned jump_to_next_tensor_B = 0,
                                         const unsigned jump_to_next_tensor_C = 0) 
{
    A += blockIdx.x * jump_to_next_tensor_A;
    B += blockIdx.x * jump_to_next_tensor_B;
    C += blockIdx.x * jump_to_next_tensor_C;

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

void device_blas_gemm(const CBLAS_LAYOUT Layout, 
                      const CBLAS_TRANSPOSE transa, 
                      const CBLAS_TRANSPOSE transb, 
                      const int m, const int n, const int k, 
                      const real alpha, const real *A, const int lda, 
                      const real *B, const int ldb, 
                      const real beta, real *C, const int ldc,
                      const unsigned jump_to_next_tensor_A,
                      const unsigned jump_to_next_tensor_B,
                      const unsigned jump_to_next_tensor_C, 
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
                                                                       alpha, A, lda,
                                                                       B, ldb, 
                                                                       beta, C, ldc,
                                                                       jump_to_next_tensor_A,
                                                                       jump_to_next_tensor_B,
                                                                       jump_to_next_tensor_C);
            CUDA_CHECK;
        }
        else if ((transa == CblasTrans) && (transb == CblasTrans)) {
            kernel_cuda_blas_gemm_TT<<<grid, block, shared_mem_size>>>(m, n, k,
                                                                       alpha, A, lda,
                                                                       B, ldb, 
                                                                       beta, C, ldc,
                                                                       jump_to_next_tensor_A,
                                                                       jump_to_next_tensor_B,
                                                                       jump_to_next_tensor_C);
            CUDA_CHECK;
        }
        else if ((transa == CblasNoTrans) && (transb == CblasTrans)) {
            kernel_cuda_blas_gemm_NT<<<grid, block, shared_mem_size>>>(m, n, k,
                                                                       alpha, A, lda,
                                                                       B, ldb, 
                                                                       beta, C, ldc,
                                                                       jump_to_next_tensor_A,
                                                                       jump_to_next_tensor_B,
                                                                       jump_to_next_tensor_C);
            CUDA_CHECK;
        }
        else if ((transa == CblasTrans) && (transb == CblasNoTrans)) {
            kernel_cuda_blas_gemm_TN<<<grid, block, shared_mem_size>>>(m, n, k,
                                                                       alpha, A, lda,
                                                                       B, ldb, 
                                                                       beta, C, ldc,
                                                                       jump_to_next_tensor_A,
                                                                       jump_to_next_tensor_B,
                                                                       jump_to_next_tensor_C);
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
