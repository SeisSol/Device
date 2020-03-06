#include <iostream>
#include <assert.h>
#include <device.h>
#include <driver_types.h>

#include "AbstractAPI.h"
#include "Internals.h"

#include "Gemm.cuh"

using namespace device;
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
void DeviceInstance::gemm(const MATRIX_LAYOUT Layout,
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

  cudaStream_t *Stream = static_cast<cudaStream_t*>(api->getRawCurrentComputeStream());

  /*
  cudaStream_t NullStream = 0;
  cudaStream_t *Stream = &NullStream;
  */
  if (Layout == ColMajor) {

    // perform initial check: similar to netlib blas
    assert((m != 0) && (n != 0) && (k != 0));
    assert(ldc >= std::max(1, m));

    const int NumRows_A = Transa == NoTrans ? m : k;
    const int NumRows_B = Transb == NoTrans ? k : n;
    assert(lda >= std::max(1, NumRows_A));
    assert(ldb >= std::max(1, NumRows_B));

    if ((Transa == NoTrans) && (Transb == NoTrans)) {
      kernel_gemmNN<<<Grid, Block, SharedMemSize, *Stream>>>(m, n, k,
          Alpha, A_base, lda,
          B_base, ldb,
          Beta, C_base, ldc,
          Offsets_A,
          Offsets_B,
          Offsets_C);
      CHECK_ERR;
    }
    else if ((Transa == Trans) && (Transb == Trans)) {
      kernel_gemmTT<<<Grid, Block, SharedMemSize, *Stream>>>(m, n, k,
          Alpha, A_base, lda,
          B_base, ldb,
          Beta, C_base, ldc,
          Offsets_A,
          Offsets_B,
          Offsets_C);
      CHECK_ERR;
    }
    else if ((Transa == NoTrans) && (Transb == Trans)) {
      kernel_gemmNT<<<Grid, Block, SharedMemSize, *Stream>>>(m, n, k,
          Alpha, A_base, lda,
          B_base, ldb,
          Beta, C_base, ldc,
          Offsets_A,
          Offsets_B,
          Offsets_C);
      CHECK_ERR;
    }
    else if ((Transa == Trans) && (Transb == NoTrans)) {
      kernel_gemmTN<<<Grid, Block, SharedMemSize, *Stream>>>(m, n, k,
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
  DeviceInstance& device = DeviceInstance::getInstance();
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