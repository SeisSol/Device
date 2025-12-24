// SPDX-FileCopyrightText: 2023-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_ALGORITHMS_COMMON_H_
#define SEISSOLDEVICE_ALGORITHMS_COMMON_H_

#include "AbstractAPI.h"
#include "Algorithms.h"
#include <limits>

#include "Internals.h"

#if defined(__ACPP__)
#include <sycl/sycl.hpp>
#endif

namespace device {
inline size_t alignToMultipleOf(size_t size, size_t base) {
  return ((size + base - 1) / base) * base;
}

template<typename F>
int blockcount(F&& func, int blocksize = internals::DefaultBlockDim);

#if defined(__ACPP__) || (defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER))
#define DEVICE_DEVICEFUNC inline
template<typename T>
DEVICE_DEVICEFUNC void ntstore(T* location, T value) {
    __builtin_nontemporal_store(value, location);
}

template<typename T>
DEVICE_DEVICEFUNC T ntload(const T* location) {
    return __builtin_nontemporal_load(location);
}

template<typename F>
int blockcount(F&& func, int blocksize) {
    return 1;
}

using LocalInt4 = __attribute__((vector_size(16))) int;
using LocalInt2 = __attribute__((vector_size(8))) int;
#elif defined(__CUDACC__) || (defined(__HIP_PLATFORM_NVIDIA__) && defined(__HIP__))
#define DEVICE_DEVICEFUNC __device__ __forceinline__

template<typename T>
DEVICE_DEVICEFUNC void ntstore(T* location, T value) {
    __stcs(location, value);
}

template<typename T>
DEVICE_DEVICEFUNC T ntload(const T* location) {
    return __ldcs(location);
}

template<typename F>
int blockcount(F&& func, int blocksize) {
    int device = 0;
    int smCount = 0;
    int blocksPerSM = 0;
    APIWRAP(cudaGetDevice(&device));
    CHECK_ERR;
    APIWRAP(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    CHECK_ERR;
    APIWRAP(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, std::forward<F>(func), blocksize, 0));
    CHECK_ERR;
    return smCount * blocksPerSM;
}

using LocalInt4 = int4;
using LocalInt2 = int2;
#elif defined(__HIP_PLATFORM_AMD__) && defined(__HIP__)
#define DEVICE_DEVICEFUNC __device__ __forceinline__
template<typename T>
DEVICE_DEVICEFUNC void ntstore(T* location, T value) {
    __builtin_nontemporal_store(value, location);
}

template<typename T>
DEVICE_DEVICEFUNC T ntload(const T* location) {
    return __builtin_nontemporal_load(location);
}

template<typename F>
int blockcount(F&& func, int blocksize) {
    int device = 0;
    int smCount = 0;
    int blocksPerSM = 0;
    APIWRAP(hipGetDevice(&device));
    CHECK_ERR;
    APIWRAP(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
    CHECK_ERR;
    APIWRAP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, std::forward<F>(func), blocksize, 0));
    CHECK_ERR;
    return smCount * blocksPerSM;
}

using LocalInt4 = __attribute__((vector_size(16))) int;
using LocalInt2 = __attribute__((vector_size(8))) int;
#else
#define DEVICE_DEVICEFUNC
template<typename T>
DEVICE_DEVICEFUNC void ntstore(T* location, T value) {
    *location = value;
}

template<typename T>
DEVICE_DEVICEFUNC T ntload(const T* location) {
    return *location;
}

template<typename F>
int blockcount(F&& func, int blocksize) {
    return 1;
}

struct LocalInt4 {int data[4];};
struct LocalInt2 {int data[2];};
#endif

template<typename T, bool OnlyFull>
DEVICE_DEVICEFUNC std::size_t iimemcpy(void* dst, const void* src, std::size_t offset, std::size_t count, int local, std::size_t stride) {
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

DEVICE_DEVICEFUNC void imemcpy(void* dst, const void* src, std::size_t count, int local, std::size_t stride) {
    std::size_t offset = 0;
    offset += iimemcpy<LocalInt4, true>(dst, src, offset, count, local, stride);
    offset += iimemcpy<LocalInt2, true>(dst, src, offset, count, local, stride);
    offset += iimemcpy<int, false>(dst, src, offset, count, local, stride);
    offset += iimemcpy<char, false>(dst, src, offset, count, local, stride);
}

template<typename T, bool OnlyFull>
DEVICE_DEVICEFUNC std::size_t iimemset(void* dst, std::size_t offset, std::size_t count, int local, std::size_t stride) {
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
    std::size_t offset = 0;
    offset += iimemset<LocalInt4, true>(dst, offset, count, local, stride);
    offset += iimemset<LocalInt2, true>(dst, offset, count, local, stride);
    offset += iimemset<int, false>(dst, offset, count, local, stride);
    offset += iimemset<char, false>(dst, offset, count, local, stride);
}

} // namespace device

#endif // SEISSOLDEVICE_ALGORITHMS_COMMON_H_

