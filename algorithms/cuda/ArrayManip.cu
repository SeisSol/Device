// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "AbstractAPI.h"
#include "LocalCommon.h"
#include "Internals.h"
#include <cassert>
#include <cstdint>
#include <device.h>

namespace device {
template <typename T> __global__ void kernel_scaleArray(T *array, T scalar, const size_t numElements) {
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
#pragma unroll 4
  for (; index < numElements; index += blockDim.x * gridDim.x) {
    ntstore(&array[index], static_cast<T>(ntload(&array[index]) * scalar));
  }
}

template <typename T> void Algorithms::scaleArray(T *devArray,
                                                  T scalar,
                                                  const size_t numElements,
                                                  void* streamPtr) {
  dim3 block(device::internals::DefaultBlockDim, 1, 1);
  dim3 grid(blockcount(kernel_scaleArray<T>), 1, 1);
  auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
  kernel_scaleArray<<<grid, block, 0, stream>>>(devArray, scalar, numElements);
  CHECK_ERR;
}
template void Algorithms::scaleArray(float *devArray, float scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(double *devArray, double scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(int *devArray, int scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(char *devArray, char scalar, const size_t numElements, void* streamPtr);

//--------------------------------------------------------------------------------------------------
template <typename T> __global__ void kernel_fillArray(T *array, T scalar, const size_t numElements) {
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
#pragma unroll 4
  for (; index < numElements; index += blockDim.x * gridDim.x) {
    ntstore(&array[index], scalar);
  }
}

template <typename T> void Algorithms::fillArray(T *devArray, const T scalar, const size_t numElements, void* streamPtr) {
  dim3 block(device::internals::DefaultBlockDim, 1, 1);
  dim3 grid(blockcount(kernel_fillArray<T>), 1, 1);
  auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
  kernel_fillArray<<<grid, block, 0, stream>>>(devArray, scalar, numElements);
  CHECK_ERR;
}
template void Algorithms::fillArray(float *devArray, float scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(double *devArray, double scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(int *devArray, int scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(unsigned *devArray, unsigned scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(char *devArray, char scalar, const size_t numElements, void* streamPtr);

//--------------------------------------------------------------------------------------------------
template <typename T>
__global__ void kernel_touchMemory(T *ptr, size_t size, bool clean) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  #pragma unroll 4
  for (; id < size; id += blockDim.x * gridDim.x) {
    if (clean) {
      ntstore(&ptr[id], T(0));
    } else {
      ntstore(&ptr[id], ntload(&ptr[id]));
    }
  }
}

template <typename T>
void Algorithms::touchMemory(T *ptr, size_t size, bool clean, void* streamPtr) {
  dim3 block(device::internals::DefaultBlockDim, 1, 1);
  dim3 grid(blockcount(kernel_touchMemory<T>), 1, 1);
  auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
  kernel_touchMemory<<<grid, block, 0, stream>>>(ptr, size, clean);
  CHECK_ERR;
}

template void Algorithms::touchMemory(float *ptr, size_t size, bool clean, void* streamPtr);
template void Algorithms::touchMemory(double *ptr, size_t size, bool clean, void* streamPtr);

//--------------------------------------------------------------------------------------------------
__global__ void kernel_incrementalAdd(
  uintptr_t* out,
  uintptr_t base,
  size_t increment,
  size_t numElements) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
#pragma unroll 4
  for (; id < numElements; id += blockDim.x * gridDim.x) {
    ntstore(&out[id], base + id * increment);
  }
}

template <typename T>
void Algorithms::incrementalAdd(
  T** out,
  T *base,
  size_t increment,
  size_t numElements,
  void* streamPtr) {

  dim3 block(device::internals::DefaultBlockDim, 1, 1);
  dim3 grid(blockcount(kernel_incrementalAdd), 1, 1);
  auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
  kernel_incrementalAdd<<<grid, block, 0, stream>>>(reinterpret_cast<uintptr_t*>(out), reinterpret_cast<uintptr_t>(base), sizeof(T) * increment, numElements);
  CHECK_ERR;
}

template void Algorithms::incrementalAdd(float** out,
  float *base,
  size_t increment,
  size_t numElements,
  void* streamPtr);
template void Algorithms::incrementalAdd(double** out,
  double *base,
  size_t increment,
  size_t numElements,
  void* streamPtr);
} // namespace device

