// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "AbstractAPI.h"
#include "Internals.h"
#include "algorithms/Common.h"

#include <cassert>
#include <device.h>

namespace device {
__global__ void kernel_streamBatchedData(const void** baseSrcPtr,
                                         void** baseDstPtr,
                                         size_t elementSize,
                                         size_t elementCount) {
  for (size_t block = blockIdx.x; block < elementCount; block += gridDim.x) {
    const void* srcElement = baseSrcPtr[block];
    void* dstElement = baseDstPtr[block];
    if (srcElement != nullptr && dstElement != nullptr) {
      imemcpy(dstElement, srcElement, elementSize, threadIdx.x, device::internals::DefaultBlockDim);
    }
  }
}

void Algorithms::streamBatchedDataI(const void** baseSrcPtr,
                                    void** baseDstPtr,
                                    size_t elementSize,
                                    size_t numElements,
                                    void* streamPtr) {
  dim3 block(device::internals::DefaultBlockDim, 1, 1);
  dim3 grid(blockcount(kernel_streamBatchedData), 1, 1);
  auto stream = reinterpret_cast<internals::DeviceStreamT>(streamPtr);
  kernel_streamBatchedData<<<grid, block, 0, stream>>>(
      baseSrcPtr, baseDstPtr, elementSize, numElements);
  CHECK_ERR;
}

//--------------------------------------------------------------------------------------------------
template <typename T>
__global__ void kernel_accumulateBatchedData(const T** baseSrcPtr,
                                             T** baseDstPtr,
                                             size_t elementSize,
                                             size_t elementCount) {
  for (size_t block = blockIdx.x; block < elementCount; block += gridDim.x) {
    const T* srcElement = baseSrcPtr[block];
    T* dstElement = baseDstPtr[block];
#pragma unroll 4
    for (int index = threadIdx.x; index < elementSize;
         index += device::internals::DefaultBlockDim) {
      ntstore(&dstElement[index], ntload(&dstElement[index]) + ntload(&srcElement[index]));
    }
  }
}

template <typename T>
void Algorithms::accumulateBatchedData(
    const T** baseSrcPtr, T** baseDstPtr, size_t elementSize, size_t numElements, void* streamPtr) {
  dim3 block(device::internals::DefaultBlockDim, 1, 1);
  dim3 grid(blockcount(kernel_accumulateBatchedData<T>), 1, 1);
  auto stream = reinterpret_cast<internals::DeviceStreamT>(streamPtr);
  kernel_accumulateBatchedData<<<grid, block, 0, stream>>>(
      baseSrcPtr, baseDstPtr, elementSize, numElements);
  CHECK_ERR;
}

template void Algorithms::accumulateBatchedData(const float** baseSrcPtr,
                                                float** baseDstPtr,
                                                size_t elementSize,
                                                size_t numElements,
                                                void* streamPtr);

template void Algorithms::accumulateBatchedData(const double** baseSrcPtr,
                                                double** baseDstPtr,
                                                size_t elementSize,
                                                size_t numElements,
                                                void* streamPtr);

//--------------------------------------------------------------------------------------------------
__global__ void
    kernel_touchBatchedMemory(void** basePtr, size_t elementSize, bool clean, size_t elementCount) {
  for (size_t block = blockIdx.x; block < elementCount; block += gridDim.x) {
    void* element = basePtr[block];
    if (element != nullptr) {
      if (clean) {
        imemset(element, elementSize, threadIdx.x, device::internals::DefaultBlockDim);
      } else {
        imemcpy(element, element, elementSize, threadIdx.x, device::internals::DefaultBlockDim);
      }
    }
  }
}

void Algorithms::touchBatchedMemoryI(
    void** basePtr, size_t elementSize, size_t numElements, bool clean, void* streamPtr) {
  dim3 block(device::internals::DefaultBlockDim, 1, 1);
  dim3 grid(blockcount(kernel_touchBatchedMemory), 1, 1);
  auto stream = reinterpret_cast<internals::DeviceStreamT>(streamPtr);
  kernel_touchBatchedMemory<<<grid, block, 0, stream>>>(basePtr, elementSize, clean, numElements);
  CHECK_ERR;
}

//--------------------------------------------------------------------------------------------------
template <typename T>
__global__ void kernel_setToValue(T** out, T value, size_t elementSize, size_t elementCount) {
  for (size_t block = blockIdx.x; block < elementCount; block += gridDim.x) {
    T* element = out[block];
#pragma unroll 4
    for (int index = threadIdx.x; index < elementSize;
         index += device::internals::DefaultBlockDim) {
      ntstore(&element[index], value);
    }
  }
}

template <typename T>
void Algorithms::setToValue(
    T** out, T value, size_t elementSize, size_t numElements, void* streamPtr) {
  dim3 block(device::internals::DefaultBlockDim, 1, 1);
  dim3 grid(blockcount(kernel_setToValue<T>), 1, 1);
  auto stream = reinterpret_cast<internals::DeviceStreamT>(streamPtr);
  kernel_setToValue<<<grid, block, 0, stream>>>(out, value, elementSize, numElements);
  CHECK_ERR;
}

template void Algorithms::setToValue(
    float** out, float value, size_t elementSize, size_t numElements, void* streamPtr);
template void Algorithms::setToValue(
    double** out, double value, size_t elementSize, size_t numElements, void* streamPtr);
template void Algorithms::setToValue(
    int** out, int value, size_t elementSize, size_t numElements, void* streamPtr);
template void Algorithms::setToValue(
    unsigned** out, unsigned value, size_t elementSize, size_t numElements, void* streamPtr);
template void Algorithms::setToValue(
    char** out, char value, size_t elementSize, size_t numElements, void* streamPtr);

//--------------------------------------------------------------------------------------------------
__global__ void kernel_copyUniformToScatter(
    const void* src, void** dst, size_t srcOffset, size_t copySize, size_t elementCount) {
  for (size_t block = blockIdx.x; block < elementCount; block += gridDim.x) {
    const void* srcElement =
        reinterpret_cast<const void*>(&reinterpret_cast<const char*>(src)[block * srcOffset]);
    void* dstElement = dst[block];
    imemcpy(dstElement, srcElement, copySize, threadIdx.x, device::internals::DefaultBlockDim);
  }
}

void Algorithms::copyUniformToScatterI(const void* src,
                                       void** dst,
                                       size_t srcOffset,
                                       size_t copySize,
                                       size_t numElements,
                                       void* streamPtr) {
  dim3 block(device::internals::DefaultBlockDim, 1, 1);
  dim3 grid(blockcount(kernel_copyUniformToScatter), 1, 1);
  auto stream = reinterpret_cast<internals::DeviceStreamT>(streamPtr);
  kernel_copyUniformToScatter<<<grid, block, 0, stream>>>(
      src, dst, srcOffset, copySize, numElements);
  CHECK_ERR;
}

//--------------------------------------------------------------------------------------------------
__global__ void kernel_copyScatterToUniform(
    const void** src, void* dst, size_t dstOffset, size_t copySize, size_t elementCount) {
  for (size_t block = blockIdx.x; block < elementCount; block += gridDim.x) {
    const void* srcElement = src[block];
    void* dstElement = reinterpret_cast<void*>(&reinterpret_cast<char*>(dst)[block * dstOffset]);
    imemcpy(dstElement, srcElement, copySize, threadIdx.x, device::internals::DefaultBlockDim);
  }
}

void Algorithms::copyScatterToUniformI(const void** src,
                                       void* dst,
                                       size_t dstOffset,
                                       size_t copySize,
                                       size_t numElements,
                                       void* streamPtr) {
  dim3 block(device::internals::DefaultBlockDim, 1, 1);
  dim3 grid(blockcount(kernel_copyScatterToUniform), 1, 1);
  auto stream = reinterpret_cast<internals::DeviceStreamT>(streamPtr);
  kernel_copyScatterToUniform<<<grid, block, 0, stream>>>(
      src, dst, dstOffset, copySize, numElements);
  CHECK_ERR;
}
} // namespace device
