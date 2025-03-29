// SPDX-FileCopyrightText: 2025 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_ALGORITHMS_CUDA_LOCALCOMMON_H_
#define SEISSOLDEVICE_ALGORITHMS_CUDA_LOCALCOMMON_H_

#include "Internals.h"

namespace device {

#if defined(__HIP_PLATFORM_AMD__) && defined(__HIP__)
#define cudaGetDevice hipGetDevice
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#endif

template<typename F>
int blockcount(F&& func, int blocksize = internals::DefaultBlockDim) {
    int device = 0;
    int smCount = 0;
    int blocksPerSM = 0;
    cudaGetDevice(&device);
    CHECK_ERR;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    CHECK_ERR;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, std::forward<F>(func), blocksize, 0);
    CHECK_ERR;
    return smCount * blocksPerSM;
}

#if defined(__CUDACC__) || (defined(__HIP_PLATFORM_NVIDIA__) && defined(__HIP__))
template<typename T>
__device__ __forceinline__ void ntstore(T* location, T value) {
    __stcs(location, value);
}

template<typename T>
__device__ __forceinline__ T ntload(const T* location) {
    return __ldcs(location);
}
#endif

#if defined(__HIP_PLATFORM_AMD__) && defined(__HIP__)
template<typename T>
__device__ __forceinline__ void ntstore(T* location, T value) {
    __builtin_nontemporal_store(value, location);
}

template<typename T>
__device__ __forceinline__ T ntload(const T* location) {
    return __builtin_nontemporal_load(location);
}
#endif

} // namespace device


#endif // SEISSOLDEVICE_ALGORITHMS_CUDA_LOCALCOMMON_H_

