// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_ALGORITHMS_CUDA_REDUCTION_H_
#define SEISSOLDEVICE_ALGORITHMS_CUDA_REDUCTION_H_

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


#endif // SEISSOLDEVICE_ALGORITHMS_CUDA_REDUCTION_H_

