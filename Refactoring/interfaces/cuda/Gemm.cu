#include <assert.h>
#include <iostream>

#include "AbstractAPI.h"
#include "Internals.h"
#include "ComputeAddressing.h"

#include <device.h>

using namespace device;

//--------------------------------------------------------------------------------------------------
// (Transa == NoTrans) && (Transb == NoTrans)
template <typename AT, typename BT, typename CT>
__global__ void kernel_gemmNN(const int m, const int n, const int k,
                              const real Alpha, AT A, const int lda,
                              BT B, const int ldb,
                              const real Beta, CT C, const int ldc,
                              unsigned Offsets_A = 0,
                              unsigned Offsets_B = 0,
                              unsigned Offsets_C = 0) {

  const real *Matrix_A = device::addressing::findData(A, Offsets_A, blockIdx.x);
  const real *Matrix_B = device::addressing::findData(B, Offsets_B, blockIdx.x);
  real *Matrix_C = device::addressing::findData(C, Offsets_C, blockIdx.x);

  // declare a pointer to array Matrix_A within shared memery
  extern __shared__ real sm_A[];

  // declare a pointer to array Matrix_B within shared memery
  // NOTE: sm_B and sm_A points into the same chunch of memory
  //       However, sm_B is shifted from sm_A into m * k elements
  real *sm_B = sm_A + m * k;

  // load matrix Matrix_A to shared memory
  // slide along matrix columns
  const unsigned ColumnRepeats = (k + n - 1) / n;
  for (int ColumnShift = 0; ColumnShift < ColumnRepeats; ++ColumnShift) {

    int Column = threadIdx.y + ColumnShift * blockDim.y;
    if (Column < k)
      sm_A[threadIdx.x + Column * m] = Matrix_A[threadIdx.x + Column * lda];
  }

  // load matrix Matrix_B to shared memory
  // slide along matrix rows
  int RowRepeats = (k + m - 1) / m;
  for (int RowShift = 0; RowShift < RowRepeats; ++RowShift) {
    int Row = threadIdx.x + RowShift * blockDim.x;
    if (Row < k)
      sm_B[Row + threadIdx.y * k] = Matrix_B[Row + threadIdx.y * ldb];
  }

  __syncthreads();

  real Result = 0.0;
  for (int j = 0; j < k; ++j) {
    Result += (sm_A[threadIdx.x + j * m] * sm_B[j + threadIdx.y * k]);
  }

  Matrix_C[threadIdx.x + threadIdx.y * ldc] = Alpha * Result
                                            + Beta * Matrix_C[threadIdx.x + threadIdx.y * ldc];
}
//--------------------------------------------------------------------------------------------------
// (transa == CblasTrans) && (transb == CblasTrans)
template <typename AT, typename BT, typename CT>
__global__ void kernel_gemmTT(const int m, const int n, const int k,
                                         const real Alpha, AT A, const int lda,
                                         BT B, const int ldb,
                                         const real Beta, CT C, const int ldc,
                                         unsigned Offsets_A = 0,
                                         unsigned Offsets_B = 0,
                                         unsigned Offsets_C = 0) {

  const real *Matrix_A = device::addressing::findData(A, Offsets_A, blockIdx.x);
  const real *Matrix_B = device::addressing::findData(B, Offsets_B, blockIdx.x);
  real *Matrix_C = device::addressing::findData(C, Offsets_C, blockIdx.x);

  // declare a pointer to array Matrix_A within shared memery
  extern __shared__ real sm_A[];

  // declare a pointer to array Matrix_B within shared memery
  // NOTE: sm_B and sm_A points into the same chunch of memory
  //       However, sm_B is shifted from sm_A into m * k elements
  real *sm_B = sm_A + m * k;

  // load matrix Matrix_A to shared memory
  // slide along matrix columns
  int ColumnRepeats = (m + n - 1) / n;
  for (int ColumnShift = 0; ColumnShift < ColumnRepeats; ++ColumnShift) {
    int Column = threadIdx.y + ColumnShift * blockDim.y;

    // slide along matrix rows
    int RowRepeats = (k + m - 1) / m;
    for (int RowShift = 0; RowShift < RowRepeats; ++RowShift) {
      int Row = threadIdx.x + RowShift * blockDim.x;
      if ((Row < k) && (Column < m)) {
        sm_A[Row + Column * k] = Matrix_A[Row + Column * lda];
      }
    }
  }

  // load matrix Matrix_B to shared memory
  // slide along matrix columns
  ColumnRepeats = (k + n - 1) / n;
  for (int ColumnShift = 0; ColumnShift < ColumnRepeats; ++ColumnShift) {
    int Column = threadIdx.y + ColumnShift * blockDim.y;

    // slide along matrix rows
    int RowRepeats = (n + m - 1) / m;
    for (int RowShift = 0; RowShift < RowRepeats; ++RowShift) {
      int Row = threadIdx.x + RowShift * blockDim.x;
      if ((Row < n) && (Column < k)) {
        sm_B[Row + Column * n] = Matrix_B[Row + Column * ldb];
      }
    }
  }

  __syncthreads();

  real Result = 0.0;
  for (int j = 0; j < k; ++j) {
    Result += (sm_A[j + threadIdx.x * k] * sm_B[threadIdx.y + j * n]);
  }

  Matrix_C[threadIdx.x + threadIdx.y * ldc] = Alpha * Result
                                            + Beta * Matrix_C[threadIdx.x + threadIdx.y * ldc];
}
//--------------------------------------------------------------------------------------------------
// (transa == CblasNoTrans) && (transb == CblasTrans)
template <typename AT, typename BT, typename CT>
__global__ void kernel_gemmNT(const int m, const int n, const int k,
                              const real Alpha, AT A, const int lda,
                              BT B, const int ldb,
                              const real Beta, CT C, const int ldc,
                              unsigned Offsets_A = 0,
                              unsigned Offsets_B = 0,
                              unsigned Offsets_C = 0) {

  const real *Matrix_A = device::addressing::findData(A, Offsets_A, blockIdx.x);
  const real *Matrix_B = device::addressing::findData(B, Offsets_B, blockIdx.x);
  real *Matrix_C = device::addressing::findData(C, Offsets_C, blockIdx.x);

  // declare a pointer to array Matrix_A within shared memery
  extern __shared__ real sm_A[];

  // declare a pointer to array Matrix_B within shared memery
  // NOTE: sm_B and sm_A points into the same chunch of memory
  //       However, sm_B is shifted from sm_A into m * k elements
  real *sm_B = sm_A + m * k;

  // load matrix Matrix_A to shared memory
  // slide along matrix columns
  int ColumnRepeats = (k + n - 1) / n;
  for (int ColumnShift = 0; ColumnShift < ColumnRepeats; ++ColumnShift) {
    int Column = threadIdx.y + ColumnShift * blockDim.y;
    if (Column < k)
      sm_A[threadIdx.x + Column * m] = Matrix_A[threadIdx.x + Column * lda];
  }


  // load matrix Matrix_B to shared memory
  // slide along matrix columns
  ColumnRepeats = (k + n - 1) / n;
  for (int ColumnShift = 0; ColumnShift < ColumnRepeats; ++ColumnShift) {
    int Column = threadIdx.y + ColumnShift * blockDim.y;

    // slide along matrix rows
    int RowRepeats = (n + m - 1) / m;
    for (int RowShift = 0; RowShift < RowRepeats; ++RowShift) {
      int Row = threadIdx.x + RowShift * blockDim.x;
      if ((Row < n) && (Column < k)) {
        sm_B[Row + Column * n] = Matrix_B[Row + Column * ldb];
      }
    }
  }

  __syncthreads();

  real Result = 0.0;
  for (int j = 0; j < k; ++j) {
    Result += (sm_A[threadIdx.x + j * m] * sm_B[threadIdx.y + j * n]);
  }


  Matrix_C[threadIdx.x + threadIdx.y * ldc] = Alpha * Result
                                            + Beta * Matrix_C[threadIdx.x + threadIdx.y * ldc];
}
//--------------------------------------------------------------------------------------------------
// (transa == CblasTrans) && (transb == CblasNoTrans)
template <typename AT, typename BT, typename CT>
__global__ void kernel_gemmTN(const int m, const int n, const int k,
                              const real Alpha, AT A, const int lda,
                              BT B, const int ldb,
                              const real Beta, CT C, const int ldc,
                              unsigned Offsets_A = 0,
                              unsigned Offsets_B = 0,
                              unsigned Offsets_C = 0) {

  const real *Matrix_A = device::addressing::findData(A, Offsets_A, blockIdx.x);
  const real *Matrix_B = device::addressing::findData(B, Offsets_B, blockIdx.x);
  real *Matrix_C = device::addressing::findData(C, Offsets_C, blockIdx.x);

  // declare a pointer to array Matrix_A within shared memery
  extern __shared__ real sm_A[];

  // declare a pointer to array Matrix_B within shared memery
  // NOTE: sm_B and sm_A points into the same chunch of memory
  //       However, sm_B is shifted from sm_A into m * k elements
  real *sm_B = sm_A + m * k;

  // load matrix Matrix_A to shared memory
  // slide along matrix columns
  int ColumnRepeats = (m + n - 1) / n;
  for (int ColumnShift = 0; ColumnShift < ColumnRepeats; ++ColumnShift) {
    int Column = threadIdx.y + ColumnShift * blockDim.y;

    // slide along matrix rows
    int RowRepeats = (k + m - 1) / m;
    for (int RowShift = 0; RowShift < RowRepeats; ++RowShift) {
      int Row = threadIdx.x + RowShift * blockDim.x;
      if ((Row < k) && (Column < m)) {
        sm_A[Row + Column * k] = Matrix_A[Row + Column * lda];
      }
    }
  }

  // load matrix Matrix_B to shared memory
  // slide along matrix rows
  int RowRepeats = (k + m - 1) / m;
  for (int RowShift = 0; RowShift < RowRepeats; ++RowShift) {
    int Row = threadIdx.x + RowShift * blockDim.x;
    if (Row < k)
      sm_B[Row + threadIdx.y * k] = Matrix_B[Row + threadIdx.y * ldb];
  }

  __syncthreads();

  real Result = 0.0;
  for (int j = 0; j < k; ++j) {
    Result += (sm_A[j + threadIdx.x * k] * sm_B[j + threadIdx.y * k]);
  }

  Matrix_C[threadIdx.x + threadIdx.y * ldc] = Alpha * Result
                                            + Beta * Matrix_C[threadIdx.x + threadIdx.y * ldc];
}


/**
 *  C = Alpha * A * B + Beta * C
 *
 *    C                A        B              C
 *
 *   _n_             __k__     _n_            _n_
 *  |   |           |     |   |   |          |   |
 *  | C | = Alpha * |m    | x |k  | + Beta * |   |
 *  |   |           |     |   |   |          |   |
 *  |m  |           |     |   |___|          |m  |
 *  |   |           |     |                  |   |
 *  |___|           |_____|                  |___|
  * */
template <typename AT, typename BT, typename CT>
void Device::gemm(const MATRIX_LAYOUT Layout,
                  const MATRIX_TRANSPOSE Transa,
                  const MATRIX_TRANSPOSE Transb,
                  const int m, const int n, const int k,
                  const real Alpha, AT A_base, const int lda,
                  BT B_base, const int ldb,
                  const real Beta, CT C_base, const int ldc,
                  unsigned Offsets_A,
                  unsigned Offsets_B,
                  unsigned Offsets_C,
                  const unsigned NumElements) {
  dim3 Block(m, n, 1);
  dim3 Grid (NumElements, 1, 1);
  size_t SharedMemSize = (m * k + k * n) * sizeof(real);

  assert((m * n) <= m_MaxBlockSize);
  assert(SharedMemSize <= m_MaxSharedMemSize);

  if (Layout == ColMajor) {

    // perform initial check: similar to netlib blas
    assert((m != 0) && (n != 0) && (k != 0));
    assert(ldc >= std::max(1, m));

    const int NumRows_A = Transa == NoTrans ? m : k;
    const int NumRows_B = Transb == NoTrans ? k : n;
    assert(lda >= std::max(1, NumRows_A));
    assert(ldb >= std::max(1, NumRows_B));

    if ((Transa == NoTrans) && (Transb == NoTrans)) {
      kernel_gemmNN<<<Grid, Block, SharedMemSize, 0>>>(m, n, k,
                                                       Alpha, A_base, lda,
                                                       B_base, ldb,
                                                       Beta, C_base, ldc,
                                                       Offsets_A,
                                                       Offsets_B,
                                                       Offsets_C);
      CHECK_ERR;
    }
    else if ((Transa == Trans) && (Transb == Trans)) {
      kernel_gemmTT<<<Grid, Block, SharedMemSize, 0>>>(m, n, k,
                                                       Alpha, A_base, lda,
                                                       B_base, ldb,
                                                       Beta, C_base, ldc,
                                                       Offsets_A,
                                                       Offsets_B,
                                                       Offsets_C);
      CHECK_ERR;
    }
    else if ((Transa == NoTrans) && (Transb == Trans)) {
      kernel_gemmNT<<<Grid, Block, SharedMemSize, 0>>>(m, n, k,
                                                       Alpha, A_base, lda,
                                                       B_base, ldb,
                                                       Beta, C_base, ldc,
                                                       Offsets_A,
                                                       Offsets_B,
                                                       Offsets_C);
      CHECK_ERR;
    }
    else if ((Transa == Trans) && (Transb == NoTrans)) {
      kernel_gemmTN<<<Grid, Block, SharedMemSize, 0>>>(m, n, k,
                                                       Alpha, A_base, lda,
                                                       B_base, ldb,
                                                       Beta, C_base, ldc,
                                                       Offsets_A,
                                                       Offsets_B,
                                                       Offsets_C);
      CHECK_ERR;
    }
  }
  else {
    std::cout << "DEVICE::ERROR the library doesn't row major gemm operation" << std::endl;
    exit(EXIT_SUCCESS);
  }
}


#define gemm(IN1, IN2, OUT) device.gemm(ColMajor, Trans, Trans, 0, 0, 0, 0.0, IN1, 0, IN2, 0, 0.0, OUT, 0, 0, 0, 0, 0)

#pragma push
#pragma O0
void instantiateGemmTemplates() {
  Device& device = Device::getInstance();
  real const ** zero{}; real const* one{}; real** two{}; real * three{};
#pragma noinline
  gemm(zero, zero ,two);
#pragma noinline
  gemm(zero, zero, three);
#pragma noinline
  gemm(zero, one, two);
#pragma noinline
  gemm(zero, one, three);
#pragma noinline
  gemm(zero, two, two);
#pragma noinline
  gemm(zero, two, three);
#pragma noinline
  gemm(zero, three, two);
#pragma noinline
  gemm(zero, three, three);
#pragma noinline
  gemm(one, zero, two);
#pragma noinline
  gemm(one, zero, three);
#pragma noinline
  gemm(one, one, two);
#pragma noinline
  gemm(one, one, three);
#pragma noinline
  gemm(one, two, two);
#pragma noinline
  gemm(one, two, three);
#pragma noinline
  gemm(one, three, two);
#pragma noinline
  gemm(one, three, three);
#pragma noinline
  gemm(two, zero, two);
#pragma noinline
  gemm(two, zero, three);
#pragma noinline
  gemm(two, one, two);
#pragma noinline
  gemm(two, one, three);
#pragma noinline
  gemm(two, two, two);
#pragma noinline
  gemm(two, two, three);
#pragma noinline
  gemm(two, three, two);
#pragma noinline
  gemm(two, three, three);
#pragma noinline
  gemm(three, zero, two);
#pragma noinline
  gemm(three, zero, three);
#pragma noinline
  gemm(three, one, two);
#pragma noinline
  gemm(three, one, three);
#pragma noinline
  gemm(three, two, two);
#pragma noinline
  gemm(three, two, three);
#pragma noinline
  gemm(three, three, two);
#pragma noinline
  gemm(three, three, three);
}
#pragma pop