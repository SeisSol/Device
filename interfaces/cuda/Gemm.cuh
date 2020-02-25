#include "ComputeAddressing.h"

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

  const real *Matrix_A = addressing::findData(A, Offsets_A, blockIdx.x);
  const real *Matrix_B = addressing::findData(B, Offsets_B, blockIdx.x);
  real *Matrix_C = addressing::findData(C, Offsets_C, blockIdx.x);

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

  const real *Matrix_A = addressing::findData(A, Offsets_A, blockIdx.x);
  const real *Matrix_B = addressing::findData(B, Offsets_B, blockIdx.x);
  real *Matrix_C = addressing::findData(C, Offsets_C, blockIdx.x);

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

  const real *Matrix_A = addressing::findData(A, Offsets_A, blockIdx.x);
  const real *Matrix_B = addressing::findData(B, Offsets_B, blockIdx.x);
  real *Matrix_C = addressing::findData(C, Offsets_C, blockIdx.x);

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

  const real *Matrix_A = addressing::findData(A, Offsets_A, blockIdx.x);
  const real *Matrix_B = addressing::findData(B, Offsets_B, blockIdx.x);
  real *Matrix_C = addressing::findData(C, Offsets_C, blockIdx.x);

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