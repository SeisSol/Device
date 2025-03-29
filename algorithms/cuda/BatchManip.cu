// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "AbstractAPI.h"
#include "LocalCommon.h"
#include "Internals.h"
#include <device.h>
#include <cassert>

namespace device {
  template<typename T>
  __global__ void kernel_streamBatchedData(T **baseSrcPtr,
                                           T **baseDstPtr,
                                           size_t elementSize, size_t elementCount) {
    for (size_t block = blockIdx.x; block < elementCount; block += gridDim.x) {
      T *srcElement = baseSrcPtr[block];
      T *dstElement = baseDstPtr[block];
#pragma unroll 4
      for (int index = threadIdx.x; index < elementSize; index += device::internals::DefaultBlockDim) {
        ntstore(&dstElement[index], ntload(&srcElement[index]));
      }
    }
  }

  template<typename T>
  void Algorithms::streamBatchedData(T **baseSrcPtr,
                                     T **baseDstPtr,
                                     unsigned elementSize,
                                     unsigned numElements,
                                     void* streamPtr) {
    dim3 block(device::internals::DefaultBlockDim, 1, 1);
    dim3 grid(blockcount(kernel_streamBatchedData<T>), 1, 1);
    auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
    kernel_streamBatchedData<<<grid, block, 0, stream>>>(baseSrcPtr, baseDstPtr, elementSize, numElements);
    CHECK_ERR;
  }

  template void Algorithms::streamBatchedData(float **baseSrcPtr,
                                     float **baseDstPtr,
                                     unsigned elementSize,
                                     unsigned numElements,
                                     void* streamPtr);
                                     
  template void Algorithms::streamBatchedData(double **baseSrcPtr,
                                     double **baseDstPtr,
                                     unsigned elementSize,
                                     unsigned numElements,
                                     void* streamPtr);


//--------------------------------------------------------------------------------------------------
  template<typename T>
  __global__ void kernel_accumulateBatchedData(T **baseSrcPtr,
                                               T **baseDstPtr,
                                               size_t elementSize, size_t elementCount) {
    for (size_t block = blockIdx.x; block < elementCount; block += gridDim.x) {
      T *srcElement = baseSrcPtr[block];
      T *dstElement = baseDstPtr[block];
  #pragma unroll 4
      for (int index = threadIdx.x; index < elementSize; index += device::internals::DefaultBlockDim) {
        ntstore(&dstElement[index], ntload(&dstElement[index]) + ntload(&srcElement[index]));
      }
    }
  }

  template<typename T>
  void Algorithms::accumulateBatchedData(T **baseSrcPtr,
                                         T **baseDstPtr,
                                         unsigned elementSize,
                                         unsigned numElements,
                                         void* streamPtr) {
    dim3 block(device::internals::DefaultBlockDim, 1, 1);
    dim3 grid(blockcount(kernel_accumulateBatchedData<T>), 1, 1);
    auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
    kernel_accumulateBatchedData<<<grid, block, 0, stream>>>(baseSrcPtr, baseDstPtr, elementSize, numElements);
    CHECK_ERR;
  }

  template void Algorithms::accumulateBatchedData(float **baseSrcPtr,
                                         float **baseDstPtr,
                                         unsigned elementSize,
                                         unsigned numElements,
                                         void* streamPtr);

  template void Algorithms::accumulateBatchedData(double **baseSrcPtr,
                                         double **baseDstPtr,
                                         unsigned elementSize,
                                         unsigned numElements,
                                         void* streamPtr);

//--------------------------------------------------------------------------------------------------
  template<typename T>
  __global__ void kernel_touchBatchedMemory(T **basePtr, unsigned elementSize, bool clean, size_t elementCount) {
    for (size_t block = blockIdx.x; block < elementCount; block += gridDim.x) {
      T *element = basePtr[block];
      if (element != nullptr) {
  #pragma unroll 4
        for (int index = threadIdx.x; index < elementSize; index += device::internals::DefaultBlockDim) {
          if (clean) {
            ntstore(&element[index], T(0));
          } else {
            ntstore(&element[index], ntload(&element[index]));
          }
        }
      }
    }
  }

  template<typename T>
  void Algorithms::touchBatchedMemory(T **basePtr,
                                      unsigned elementSize,
                                      unsigned numElements,
                                      bool clean,
                                      void* streamPtr) {
    dim3 block(device::internals::DefaultBlockDim, 1, 1);
    dim3 grid(blockcount(kernel_touchBatchedMemory<T>), 1, 1);
    auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
    kernel_touchBatchedMemory<<<grid, block, 0, stream>>>(basePtr, elementSize, clean, numElements);
    CHECK_ERR;
  }

  template void Algorithms::touchBatchedMemory(float **basePtr,
                                      unsigned elementSize,
                                      unsigned numElements,
                                      bool clean,
                                      void* streamPtr);

  template void Algorithms::touchBatchedMemory(double **basePtr,
                                      unsigned elementSize,
                                      unsigned numElements,
                                      bool clean,
                                      void* streamPtr);


//--------------------------------------------------------------------------------------------------
template<typename T>
__global__  void kernel_setToValue(T** out, T value, size_t elementSize, size_t elementCount) {
  for (size_t block = blockIdx.x; block < elementCount; block += gridDim.x) {
    T *element = out[block];
#pragma unroll 4
    for (int index = threadIdx.x; index < elementSize; index += device::internals::DefaultBlockDim) {
      ntstore(&element[index], value);
    }
  }
}

template<typename T>
void Algorithms::setToValue(T** out, T value, size_t elementSize, size_t numElements, void* streamPtr) {
  dim3 block(device::internals::DefaultBlockDim, 1, 1);
  dim3 grid(blockcount(kernel_setToValue<T>), 1, 1);
  auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
  kernel_setToValue<<<grid, block, 0, stream>>>(out, value, elementSize, numElements);
  CHECK_ERR;
}

template void Algorithms::setToValue(float** out, float value, size_t elementSize, size_t numElements, void* streamPtr);
template void Algorithms::setToValue(double** out, double value, size_t elementSize, size_t numElements, void* streamPtr);

//--------------------------------------------------------------------------------------------------
  template<typename T>
  __global__ void kernel_copyUniformToScatter(T *src, T **dst, size_t srcOffset, size_t copySize, size_t elementCount) {
    for (size_t block = blockIdx.x; block < elementCount; block += gridDim.x) {
      T *srcElement = &src[block * srcOffset];
      T *dstElement = dst[block];
  #pragma unroll 4
      for (int index = threadIdx.x; index < copySize; index += device::internals::DefaultBlockDim) {
        ntstore(&dstElement[index], ntload(&srcElement[index]));
      }
    }
  }

  template<typename T>
  void Algorithms::copyUniformToScatter(T *src,
                                        T **dst,
                                        size_t srcOffset,
                                        size_t copySize,
                                        size_t numElements,
                                        void* streamPtr) {
    dim3 block(device::internals::DefaultBlockDim, 1, 1);
    dim3 grid(blockcount(kernel_copyUniformToScatter<T>), 1, 1);
    auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
    kernel_copyUniformToScatter<<<grid, block, 0, stream>>>(src, dst, srcOffset, copySize, numElements);
    CHECK_ERR;
  }
  template void Algorithms::copyUniformToScatter(double *src,
                                                 double **dst,
                                                 size_t srcOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);

  template void Algorithms::copyUniformToScatter(float *src,
                                                 float **dst,
                                                 size_t srcOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);

  template void Algorithms::copyUniformToScatter(int *src,
                                                 int **dst,
                                                 size_t srcOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);

  template void Algorithms::copyUniformToScatter(char *src,
                                                 char **dst,
                                                 size_t srcOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);

//--------------------------------------------------------------------------------------------------
  template<typename T>
  __global__ void kernel_copyScatterToUniform(T **src, T *dst, size_t dstOffset, size_t copySize, size_t elementCount) {
    for (size_t block = blockIdx.x; block < elementCount; block += gridDim.x) {
      T *srcElement = src[block];
      T *dstElement = &dst[block * dstOffset];
  #pragma unroll 4
      for (int index = threadIdx.x; index < copySize; index += device::internals::DefaultBlockDim) {
        ntstore(&dstElement[index], ntload(&srcElement[index]));
      }
    }
  }

  template<typename T>
  void Algorithms::copyScatterToUniform(T **src,
                                        T *dst,
                                        size_t dstOffset,
                                        size_t copySize,
                                        size_t numElements,
                                        void* streamPtr) {
    dim3 block(device::internals::DefaultBlockDim, 1, 1);
    dim3 grid(blockcount(kernel_copyScatterToUniform<T>), 1, 1);
    auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
    kernel_copyScatterToUniform<<<grid, block, 0, stream>>>(src, dst, dstOffset, copySize, numElements);
    CHECK_ERR;
  }
  template void Algorithms::copyScatterToUniform(double **src,
                                                 double *dst,
                                                 size_t dstOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);

  template void Algorithms::copyScatterToUniform(float **src,
                                                 float *dst,
                                                 size_t dstOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);

  template void Algorithms::copyScatterToUniform(int **src,
                                                 int *dst,
                                                 size_t dstOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);

  template void Algorithms::copyScatterToUniform(char **src,
                                                 char *dst,
                                                 size_t dstOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);
} // namespace device

