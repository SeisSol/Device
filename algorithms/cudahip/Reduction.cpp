// SPDX-FileCopyrightText: 2021 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Internals.h"
#include "algorithms/Common.h"

#include <cassert>
#include <device.h>
#include <limits>
#include <math.h>
#include <string.h>
#include <type_traits>

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
  // lowest(), not min(): for floating point types min() is the smallest positive normal value,
  // which is larger than every negative input
  T defaultValue{std::numeric_limits<T>::lowest()};
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

template <std::size_t Size>
struct AtomicWord {
  static_assert(Size == 4 || Size == 8, "no atomic word of the size of the accumulator type");
};

template <>
struct AtomicWord<4> {
  using Type = unsigned int;
};

template <>
struct AtomicWord<8> {
  using Type = unsigned long long;
};

template <typename ToT, typename FromT>
__device__ __forceinline__ ToT reinterpretValue(const FromT& from) {
  static_assert(sizeof(ToT) == sizeof(FromT), "reinterpretValue requires types of equal size");
  ToT to{};
  memcpy(&to, &from, sizeof(ToT));
  return to;
}

// Fallback for the combinations without a native atomic. The compare-and-swap runs on a word of
// exactly the size of T: a wider word would read and write the memory next to the result.
template <typename T, typename OperationT>
__device__ __forceinline__ void atomicUpdateCas(T* address, T val, OperationT operation) {
  using WordT = typename AtomicWord<sizeof(T)>::Type;
  auto* wordAddress = reinterpret_cast<WordT*>(address);

  WordT old = *wordAddress;
  WordT assumed{};
  do {
    assumed = old;
    const T updated = operation(reinterpretValue<T>(assumed), val);
    old = atomicCAS(wordAddress, assumed, reinterpretValue<WordT>(updated));
  } while (assumed != old);
}

template <typename T, typename OperationT>
__device__ __forceinline__ void atomicUpdate(T* address, T val, OperationT operation) {
  if constexpr (std::is_same_v<OperationT, Sum<T>>) {
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
                  std::is_same_v<T, unsigned long long> || std::is_same_v<T, float>) {
      atomicAdd(address, val);
      return;
    } else if constexpr (std::is_integral_v<T> && sizeof(T) == sizeof(unsigned long long)) {
      // the unsigned addition wraps the same way, so it also gives the signed result
      atomicAdd(reinterpret_cast<unsigned long long*>(address),
                static_cast<unsigned long long>(val));
      return;
    } else if constexpr (std::is_same_v<T, double>) {
// mirrors the guard the toolkit puts on the declaration itself
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 600)
      atomicAdd(address, val);
      return;
#endif
    }
  }
  if constexpr (std::is_same_v<OperationT, Max<T>> &&
                (std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
                 std::is_same_v<T, unsigned long long>)) {
    atomicMax(address, val);
    return;
  }
  if constexpr (std::is_same_v<OperationT, Min<T>> &&
                (std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
                 std::is_same_v<T, unsigned long long>)) {
    atomicMin(address, val);
    return;
  }
  atomicUpdateCas(address, val, operation);
}

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
__launch_bounds__(BlockSize) void __global__
    kernel_reduce(AccT* result, const VecT* vector, size_t size, OperationT operation) {

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

  // the result is set either way, but there is nothing to reduce into it, and a grid of zero
  // blocks is not a valid launch configuration
  if (size == 0) {
    CHECK_ERR;
    return;
  }

  switch (type) {
  case ReductionType::Add: {
    kernel_reduce<<<numBlocks, BlockSize, 0, stream>>>(result, buffer, size, device::Sum<AccT>());
    break;
  }
  case ReductionType::Max: {
    kernel_reduce<<<numBlocks, BlockSize, 0, stream>>>(result, buffer, size, device::Max<AccT>());
    break;
  }
  case ReductionType::Min: {
    kernel_reduce<<<numBlocks, BlockSize, 0, stream>>>(result, buffer, size, device::Min<AccT>());
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
