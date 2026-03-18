// SPDX-FileCopyrightText: 2021 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Internals.h"
#include "algorithms/Common.h"

#include <cassert>
#include <device.h>
#include <math.h>

namespace device {

constexpr int BlockSize = 1024;
constexpr int ItemsPerThread = 4;
template <typename T>
struct Sum {
  T defaultValue{0};
  __device__ __forceinline__ T operator()(T op1, T op2) { return op1 + op2; }
};

template <typename T>
struct Max {
  T defaultValue{std::numeric_limits<T>::min()};
  __device__ __forceinline__ T operator()(T op1, T op2) { return op1 > op2 ? op1 : op2; }
};

template <typename T>
struct Min {
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

// Warp reduce operation similar to SYCL
template <typename T, typename OperationT>
__device__ __forceinline__ T warpReduce(T value, OperationT operation) {

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value = operation(value, shuffledown(value, offset));
  }
  return value;
}

// Helper function for Generic Atomic Update
// Fallback to atomicCAS-based implementation if atomic instruction is not available
// Picked from: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
template <typename T, typename OperationT>
__device__ __forceinline__ void atomicUpdate(T* address, T val, OperationT operation) {
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    T calculatedRes = operation(*(T*)&assumed, val);
    old = atomicCAS(address_as_ull, assumed, *(unsigned long long*)&calculatedRes);
  } while (assumed != old);
}

// Native atomics
template <>
__device__ __forceinline__ void
    atomicUpdate<int, device::Sum<int>>(int* address, int val, device::Sum<int> operation) {
  atomicAdd(address, val);
}
template <>
__device__ __forceinline__ void atomicUpdate<float, device::Sum<float>>(
    float* address, float val, device::Sum<float> operation) {
  atomicAdd(address, val);
}
#if __CUDA_ARCH__ >= 600
template <>
__device__ __forceinline__ void atomicUpdate<double, device::Sum<double>>(
    double* address, double val, device::Sum<double> operation) {
  atomicAdd(address, val);
}
#endif

// Block Reduce
template <typename T, typename OperationT>
__device__ __forceinline__ T blockReduce(T val, T* shmem, OperationT operation) {

  const int laneId = threadIdx.x % warpSize;
  const int warpId = threadIdx.x / warpSize;

  val = warpReduce(val, operation);
  if (laneId == 0) {
    shmem[warpId] = val;
  }
  __syncthreads();

  const int numWarps = BlockSize / warpSize;
  val = (threadIdx.x < numWarps) ? shmem[laneId] : operation.defaultValue;

  if (warpId == 0) {
    val = warpReduce(val, operation);
  }

  return val;
}

// Init Kernel to handle overrideResult safely across multiple blocks
template <typename T, typename OperationT>
__global__ void initKernel(T* result, OperationT operation) {
  if (threadIdx.x == 0) {
    *result = operation.defaultValue;
  }
}

template <typename AccT, typename VecT, typename OperationT>
__launch_bounds__(BlockSize) void __global__ kernel_reduce(
    AccT* result, const VecT* vector, size_t size, bool overrideResult, OperationT operation) {

  // Maximum block size 1024, warp size 32 so 1024/32 = 32 chosen
  // For AMD, warp size 64, 1024/64 = 16, but 32 should work with a few idle memory addresses
  __shared__ AccT shmem[32];

  AccT threadAcc = operation.defaultValue;
  size_t blockBaseIdx = blockIdx.x * (BlockSize * ItemsPerThread);
  size_t threadBaseIdx = blockBaseIdx + threadIdx.x;

#pragma unroll
  for (int i = 0; i < ItemsPerThread; i++) {
    size_t idx = threadBaseIdx + i * BlockSize;
    if (idx < size) {
      threadAcc = operation(threadAcc, static_cast<AccT>(ntload(&vector[idx])));
    }
  }

  AccT blockAcc = blockReduce<AccT, OperationT>(threadAcc, shmem, operation);

  if (threadIdx.x == 0) {
    (void)overrideResult; // to silence unused parameter warning for non-Add reductions
    atomicUpdate(result, blockAcc, operation);
  }
}

template <typename AccT, typename VecT>
void Algorithms::reduceVector(AccT* result,
                              const VecT* buffer,
                              bool overrideResult,
                              size_t size,
                              ReductionType type,
                              void* streamPtr) {
  auto* stream = reinterpret_cast<internals::DeviceStreamT>(streamPtr);

  const size_t totalItems = BlockSize * ItemsPerThread;
  const size_t numBlocks = (size + totalItems - 1) / totalItems;

  if (overrideResult) {
    switch (type) {
    case ReductionType::Add:
      initKernel<<<1, 1, 0, stream>>>(result, device::Sum<AccT>());
      break;
    case ReductionType::Max:
      initKernel<<<1, 1, 0, stream>>>(result, device::Max<AccT>());
      break;
    case ReductionType::Min:
      initKernel<<<1, 1, 0, stream>>>(result, device::Min<AccT>());
      break;
    }
  }

  switch (type) {
  case ReductionType::Add: {
    kernel_reduce<<<numBlocks, BlockSize, 0, stream>>>(
        result, buffer, size, overrideResult, device::Sum<AccT>());
    break;
  }
  case ReductionType::Max: {
    kernel_reduce<<<numBlocks, BlockSize, 0, stream>>>(
        result, buffer, size, overrideResult, device::Max<AccT>());
    break;
  }
  case ReductionType::Min: {
    kernel_reduce<<<numBlocks, BlockSize, 0, stream>>>(
        result, buffer, size, overrideResult, device::Min<AccT>());
    break;
  }
  default: {
    assert(false && "reduction type is not implemented");
  }
  }
  CHECK_ERR;
}

template void Algorithms::reduceVector(int* result,
                                       const int* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(unsigned* result,
                                       const unsigned* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(long* result,
                                       const int* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(unsigned long* result,
                                       const unsigned* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(long* result,
                                       const long* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(unsigned long* result,
                                       const unsigned long* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(long long* result,
                                       const int* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(unsigned long long* result,
                                       const unsigned* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(long long* result,
                                       const long* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(unsigned long long* result,
                                       const unsigned long* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(long long* result,
                                       const long long* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(unsigned long long* result,
                                       const unsigned long long* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(float* result,
                                       const float* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(double* result,
                                       const float* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);
template void Algorithms::reduceVector(double* result,
                                       const double* buffer,
                                       bool overrideResult,
                                       size_t size,
                                       ReductionType type,
                                       void* streamPtr);

} // namespace device
