#ifndef DEVICE_ALGORITHMS_REDUCTION_H
#define DEVICE_ALGORITHMS_REDUCTION_H

#include "AbstractAPI.h"
#include "interfaces/cuda/Internals.h"
#include <cassert>
#include <device.h>

namespace device {
template <typename T, typename OperationT>
void __global__ kernel_reduce(T *vector, const size_t size, OperationT Operation) {
  const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
  T value = (id < size) ? vector[id] : Operation.defaultValue;

  constexpr int HALF_WARP = internals::WARP_SIZE / 2;
  constexpr unsigned int fullMask = 0xffffffff;
  for (int offset = HALF_WARP; offset >= 1; offset /= 2) {
    value = Operation(value, __shfl_down_sync(fullMask, value, offset));
  }

  if (threadIdx.x == 0) {
    vector[blockIdx.x] = value;
  }
}
} // namespace device

#endif // DEVICE_ALGORITHMS_REDUCTION_H