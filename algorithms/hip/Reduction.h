#include "interfaces/hip/Internals.h"
#include "AbstractAPI.h"
#include <cassert>
#include <device.h>

namespace device {

template <typename T, typename OperationT>
void __global__ kernel_reduce(T *to, const T *from, const size_t size, OperationT Operation) {
  const size_t id = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  T value = (id < size) ? from[id] : Operation.getDefaultValue();

  constexpr unsigned int fullMask = 0xffffffff;
  for (int offset = 16; offset > 0; offset /= 2) {
    value = Operation(value, __shfl_down_sync(fullMask, value, offset));
  }

  if (threadIdx.x == 0) {
    to[blockIdx.x] = value;
  }
}
} // namespace device