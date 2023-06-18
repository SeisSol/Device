#include "AbstractAPI.h"
#include "interfaces/hip/Internals.h"
#include <cassert>
#include <device.h>

namespace device {

template <typename T, typename OperationT>
void __global__ kernel_reduce(T *vector, const size_t size, OperationT Operation) {
  const size_t id = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  T value = (id < size) ? vector[id] : Operation.defaultValue;

  constexpr int HALF_WARP = internals::WARP_SIZE / 2;
  for (int offset = HALF_WARP; offset >= 1; offset /= 2) {
#ifdef CUDA_UNDERHOOD
    constexpr unsigned int fullMask = 0xffffffff;
    value = Operation(value, __shfl_down_sync(fullMask, value, offset));
#else
    value = Operation(value, __shfl_down(value, offset));
#endif
  }

  if (hipThreadIdx_x == 0) {
    vector[hipBlockIdx_x] = value;
  }
}
} // namespace device
