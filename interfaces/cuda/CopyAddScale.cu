#include <assert.h>

#include "AbstractAPI.h"
#include "Internals.h"
#include "ComputeAddressing.h"

#include <device.h>

using namespace device;

template <typename AT, typename BT>
__global__ void kernel_copyAddScale(const int m, const int n,
                                    const real Alpha, AT A, const int lda,
                                    const real Beta, BT B, const int ldb,
                                    unsigned Offsets_A,
                                    unsigned Offsets_B) {
  const real *Element_A = addressing::findData(A, Offsets_A, blockIdx.x);
  real *Element_B = addressing::findData(B, Offsets_B, blockIdx.x);

  Element_B[threadIdx.x + threadIdx.y * ldb] = Beta * Element_B[threadIdx.x + threadIdx.y * ldb]
                                             + Alpha * Element_A[threadIdx.x + threadIdx.y * lda];
}


template <typename AT, typename BT>
void DeviceInstance::copyAddScale(const int m, const int n,
                                  const real Alpha, AT A, const int lda,
                                  const real Beta, BT B, const int ldb,
                                  unsigned Offsets_A,
                                  unsigned Offsets_B,
                                  const unsigned NumElements) {
  dim3 Block(m, n, 1);
  dim3 Grid(NumElements, 1, 1);

  assert((m * n) <= m_MaxBlockSize);
  assert((m <= lda) && (m <= ldb));
  assert((Beta == 1.0) || (Beta == 0.0) || (Beta == -1.0));

  /*
  cudaStream_t *Stream = static_cast<cudaStream_t*>(api->getRawCurrentComputeStream());
  */

  kernel_copyAddScale<<<Grid, Block, 0, 0>>>(m, n,
                                             Alpha, A, lda,
                                             Beta, B, ldb,
                                             Offsets_A,
                                             Offsets_B);

  /*
  // TODO: has to be tested before deployment
  kernel_copyAddScale<<<Grid, Block, 0, *Stream>>>(m, n,
                                                   Alpha, A, lda,
                                                   Beta, B, ldb,
                                                   offsets_A,
                                                   offsets_B);
  */
  CHECK_ERR;
}


/** The function is used by the compiler to instantiate specific copyAddScale implementations. It is not callable
 * by the client code.
 *
 * We use pragma noinline to prevent inlining during a code optimization i.e. -O2.
 * if compiler inlines a function then the corresponding template specification is not going to appear
 * as a symbol in an object file
 */
#define add_scale(IN, OUT) device.copyAddScale(0, 0, 0.0, IN, 0, 0.0, OUT, 0, 0, 0, 0)

#pragma push
#pragma O0
void instantiateCopyAddScaleTemplates() {
  DeviceInstance& device = DeviceInstance::getInstance();
  real const ** zero{}; real const* one{}; real** two{}; real * three{};
  #pragma noinline
  add_scale(zero, two);
  #pragma noinline
  add_scale(zero, three);
  #pragma noinline
  add_scale(one, two);
  #pragma noinline
  add_scale(one, three);
  #pragma noinline
  add_scale(two, two);
  #pragma noinline
  add_scale(two, three);
  #pragma noinline
  add_scale(three, two);
}
#pragma pop

