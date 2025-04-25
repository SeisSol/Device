// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Internals.h"
#include "algorithms/Common.h"
#include <cassert>
#include <device.h>
#include <math.h>

namespace device {
template <typename T> struct Sum {
  T defaultValue{0};
  __device__ __forceinline__ T operator()(T op1, T op2) { return op1 + op2; }
};

template <typename T> struct Max {
  T defaultValue{std::numeric_limits<T>::min()};
  __device__ __forceinline__ T operator()(T op1, T op2) { return op1 > op2 ? op1 : op2; }
};

template <typename T> struct Min {
  T defaultValue{std::numeric_limits<T>::max()};
  __device__ __forceinline__ T operator()(T op1, T op2) { return op1 > op2 ? op2 : op1; }
};


#if defined(__CUDACC__) || defined(__HIP_PLATFORM_NVIDIA__) && defined(__HIP__)
template <typename T>
__forceinline__ __device__ T shuffledown(T value, int offset) {
  constexpr unsigned int fullMask = 0xffffffff;
  return __shfl_down_sync(fullMask, value, offset);
}
#endif

// no OCKL reduction for now
#if defined(__HIP_PLATFORM_AMD__) && defined(__HIP__)
template <typename T>
__forceinline__ __device__ T shuffledown(T value, int offset) {
  return __shfl_down(value, offset);
}
#endif

// a rather "dumb", but general reduction kernel
// (not intended for intensive use; there's the thrust libraries instead)

template <typename T, typename OperationT>
__launch_bounds__(1024)
void __global__ kernel_reduce(T* result, const T* vector, size_t size, bool overrideResult, OperationT operation) {
  __shared__ T shmem[256];
  const auto warpCount = blockDim.x / warpSize;
  const auto currentWarp = threadIdx.x / warpSize;
  const auto threadInWarp = threadIdx.x % warpSize;
  const auto warpsNeeded = (size + warpSize - 1) / warpSize;

  auto acc = operation.defaultValue;
  
  #pragma unroll 4
  for (std::size_t i = currentWarp; i < warpsNeeded; i += warpCount) {
    const auto id = threadInWarp + i * warpSize;
    auto value = (id < size) ? ntload(&vector[id]) : operation.defaultValue;

    for (int offset = 1; offset < warpSize; offset *= 2) {
      value = operation(value, shuffledown(value, offset));
    }

    acc = operation(acc, value);
  }

  if (threadInWarp == 0) {
    shmem[currentWarp] = acc;
  }

  __syncthreads();

  if (currentWarp == 0) {
    const auto lastWarpsNeeded = (warpCount + warpSize - 1) / warpSize;
    auto lastAcc = operation.defaultValue;
    #pragma unroll 2
    for (int i = 0; i < lastWarpsNeeded; i += warpSize) {
      const auto id = threadInWarp + i * warpSize;
      auto value = (i < warpCount) ? shmem[id] : operation.defaultValue;

      for (int offset = 1; offset < warpSize; offset *= 2) {
        value = operation(value, shuffledown(value, offset));
      }

      lastAcc = operation(lastAcc, value);
    }

    if (threadIdx.x == 0) {
      if (overrideResult) {
        ntstore(result, lastAcc);
      }
      else {
        ntstore(result, operation(ntload(result), lastAcc));
      }
    }
  }
}

template <typename T> void Algorithms::reduceVector(T* result, const T *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr) {
  auto* stream = reinterpret_cast<internals::DeviceStreamT>(streamPtr);

  dim3 grid(1, 1, 1);
  dim3 block(1024, 1, 1);

  switch (type) {
  case ReductionType::Add: {
    kernel_reduce<<<grid, block, 0, stream>>>(result, buffer, size, overrideResult, device::Sum<T>());
    break;
  }
  case ReductionType::Max: {
    kernel_reduce<<<grid, block, 0, stream>>>(result, buffer, size, overrideResult, device::Max<T>());
    break;
  }
  case ReductionType::Min: {
    kernel_reduce<<<grid, block, 0, stream>>>(result, buffer, size, overrideResult, device::Min<T>());
    break;
  }
  default: {
    assert(false && "reduction type is not implemented");
    }
  }
  CHECK_ERR;
}

template void Algorithms::reduceVector(int* result, const int *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(unsigned* result, const unsigned *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(float* result, const float *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(double* result, const double *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
} // namespace device

