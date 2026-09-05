// SPDX-FileCopyrightText: 2023 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_ALGORITHMS_COMMON_H_
#define SEISSOLDEVICE_ALGORITHMS_COMMON_H_

#include "AbstractAPI.h"
#include "Algorithms.h"
#include "Internals.h"

#include <cstdint>
#include <limits>

#if defined(__ACPP__)
#include <sycl/sycl.hpp>
#endif

namespace device {
inline size_t alignToMultipleOf(size_t size, size_t base) {
  return ((size + base - 1) / base) * base;
}

template <typename F>
int blockcount(F&& func, int blocksize = internals::DefaultBlockDim);

#if defined(__ACPP__) || (defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER))
#define DEVICE_DEVICEFUNC inline
template <typename T>
DEVICE_DEVICEFUNC void ntstore(T* location, T value) {
  __builtin_nontemporal_store(value, location);
}

template <typename T>
DEVICE_DEVICEFUNC T ntload(const T* location) {
  return __builtin_nontemporal_load(location);
}

template <typename F>
int blockcount(F&& func, int blocksize) {
  return 1;
}

using LocalInt4 = __attribute__((vector_size(16))) int;
using LocalInt2 = __attribute__((vector_size(8))) int;
#elif defined(__CUDACC__) || (defined(__HIP_PLATFORM_NVIDIA__) && defined(__HIP__))
#define DEVICE_DEVICEFUNC __device__ __forceinline__

template <typename T>
DEVICE_DEVICEFUNC void ntstore(T* location, T value) {
  __stcs(location, value);
}

template <typename T>
DEVICE_DEVICEFUNC T ntload(const T* location) {
  return __ldcs(location);
}

template <typename F>
int blockcount(F&& func, int blocksize) {
  int device = 0;
  int smCount = 0;
  int blocksPerSM = 0;
  APIWRAP(cudaGetDevice(&device));
  APIWRAP(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
  APIWRAP(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocksPerSM, std::forward<F>(func), blocksize, 0));
  return smCount * blocksPerSM;
}

using LocalInt4 = int4;
using LocalInt2 = int2;
#elif defined(__HIP_PLATFORM_AMD__) && defined(__HIP__)
#define DEVICE_DEVICEFUNC __device__ __forceinline__
template <typename T>
DEVICE_DEVICEFUNC void ntstore(T* location, T value) {
  __builtin_nontemporal_store(value, location);
}

template <typename T>
DEVICE_DEVICEFUNC T ntload(const T* location) {
  return __builtin_nontemporal_load(location);
}

template <typename F>
int blockcount(F&& func, int blocksize) {
  int device = 0;
  int smCount = 0;
  int blocksPerSM = 0;
  APIWRAP(hipGetDevice(&device));
  APIWRAP(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
  APIWRAP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocksPerSM, std::forward<F>(func), blocksize, 0));
  return smCount * blocksPerSM;
}

using LocalInt4 = __attribute__((vector_size(16))) int;
using LocalInt2 = __attribute__((vector_size(8))) int;
#else
#define DEVICE_DEVICEFUNC
template <typename T>
DEVICE_DEVICEFUNC void ntstore(T* location, T value) {
  *location = value;
}

template <typename T>
DEVICE_DEVICEFUNC T ntload(const T* location) {
  return *location;
}

template <typename F>
int blockcount(F&& func, int blocksize) {
  return 1;
}

struct LocalInt4 {
  int data[4];
};
struct LocalInt2 {
  int data[2];
};
#endif

template <typename T, bool OnlyFull>
DEVICE_DEVICEFUNC std::size_t iimemcpy(void* dst,
                                       const void* src,
                                       std::size_t offset,
                                       std::size_t count,
                                       int local,
                                       std::size_t stride) {
  T* cdst = reinterpret_cast<T*>(dst);
  const T* csrc = reinterpret_cast<const T*>(src);
  const auto start = offset / sizeof(T);
  const auto end1 = (count - offset) / sizeof(T);
  const auto end = OnlyFull ? ((end1 / stride) * stride) : end1;
#pragma unroll 4
  for (std::size_t i = local; i < end; i += stride) {
    const auto data = ntload<T>(reinterpret_cast<const T*>(csrc + start + i));
    ntstore<T>(reinterpret_cast<T*>(cdst + start + i), data);
  }
  return end * sizeof(T);
}

/**
 * Returns the address bits that keep the given pointers from being 16-byte aligned; zero means
 * both of them are. A null pointer contributes nothing, which is how the single-pointer case is
 * asked.
 *
 * The copy and fill routines below step down from 16-byte accesses to single bytes. Batched
 * buffers are addressed through a pointer table and an element stride, so their elements are not
 * guaranteed to sit on a 16-byte boundary, and a vector access to an address that is not aligned
 * to its own width faults.
 */
DEVICE_DEVICEFUNC std::size_t unalignedBits(const void* first, const void* second) {
  const auto bits =
      reinterpret_cast<std::uintptr_t>(first) | reinterpret_cast<std::uintptr_t>(second);
  return static_cast<std::size_t>(bits & 15U);
}

DEVICE_DEVICEFUNC void
    imemcpy(void* dst, const void* src, std::size_t count, int local, std::size_t stride) {
  const auto lowBits = unalignedBits(dst, src);
  std::size_t offset = 0;
  if (lowBits % sizeof(LocalInt4) == 0) {
    offset += iimemcpy<LocalInt4, true>(dst, src, offset, count, local, stride);
  }
  if (lowBits % sizeof(LocalInt2) == 0) {
    offset += iimemcpy<LocalInt2, true>(dst, src, offset, count, local, stride);
  }
  if (lowBits % sizeof(int) == 0) {
    offset += iimemcpy<int, false>(dst, src, offset, count, local, stride);
  }
  offset += iimemcpy<char, false>(dst, src, offset, count, local, stride);
}

template <typename T, bool OnlyFull>
DEVICE_DEVICEFUNC std::size_t
    iimemset(void* dst, std::size_t offset, std::size_t count, int local, std::size_t stride) {
  T* cdst = reinterpret_cast<T*>(dst);
  const auto start = offset / sizeof(T);
  const auto end1 = (count - offset) / sizeof(T);
  const auto end = OnlyFull ? ((end1 / stride) * stride) : end1;
#pragma unroll 4
  for (std::size_t i = local; i < end; i += stride) {
    const T data{};
    ntstore<T>(reinterpret_cast<T*>(cdst + start + i), data);
  }
  return end * sizeof(T);
}

DEVICE_DEVICEFUNC void imemset(void* dst, std::size_t count, int local, std::size_t stride) {
  const auto lowBits = unalignedBits(dst, nullptr);
  std::size_t offset = 0;
  if (lowBits % sizeof(LocalInt4) == 0) {
    offset += iimemset<LocalInt4, true>(dst, offset, count, local, stride);
  }
  if (lowBits % sizeof(LocalInt2) == 0) {
    offset += iimemset<LocalInt2, true>(dst, offset, count, local, stride);
  }
  if (lowBits % sizeof(int) == 0) {
    offset += iimemset<int, false>(dst, offset, count, local, stride);
  }
  offset += iimemset<char, false>(dst, offset, count, local, stride);
}

} // namespace device

#endif // SEISSOLDEVICE_ALGORITHMS_COMMON_H_
