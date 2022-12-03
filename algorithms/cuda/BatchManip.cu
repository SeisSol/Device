#include "AbstractAPI.h"
#include "interfaces/cuda/Internals.h"
#include <device.h>
#include <cassert>

namespace device {
  __global__ void kernel_streamBatchedData(real **baseSrcPtr,
                                           real **baseDstPtr,
                                           unsigned elementSize) {

    real *srcElement = baseSrcPtr[blockIdx.x];
    real *dstElement = baseDstPtr[blockIdx.x];
    for (int index = threadIdx.x; index < elementSize; index += blockDim.x) {
      dstElement[index] = srcElement[index];
    }
  }

  void Algorithms::streamBatchedData(real **baseSrcPtr,
                                     real **baseDstPtr,
                                     unsigned elementSize,
                                     unsigned numElements,
                                     void* streamPtr) {
    dim3 block(internals::WARP_SIZE, 1, 1);
    dim3 grid(numElements, 1, 1);
    auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
    kernel_streamBatchedData<<<grid, block, 0, stream>>>(baseSrcPtr, baseDstPtr, elementSize); CHECK_ERR;
  }


//--------------------------------------------------------------------------------------------------
  __global__ void kernel_accumulateBatchedData(real **baseSrcPtr,
                                               real **baseDstPtr,
                                               unsigned elementSize) {

    real *srcElement = baseSrcPtr[blockIdx.x];
    real *dstElement = baseDstPtr[blockIdx.x];
    for (int index = threadIdx.x; index < elementSize; index += blockDim.x) {
      dstElement[index] += srcElement[index];
    }
  }

  void Algorithms::accumulateBatchedData(real **baseSrcPtr,
                                         real **baseDstPtr,
                                         unsigned elementSize,
                                         unsigned numElements,
                                         void* streamPtr) {
    dim3 block(internals::WARP_SIZE, 1, 1);
    dim3 grid(numElements, 1, 1);
    auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
    kernel_accumulateBatchedData<<<grid, block, 0, stream>>>(baseSrcPtr, baseDstPtr, elementSize); CHECK_ERR;
  }

//--------------------------------------------------------------------------------------------------
  __global__ void kernel_touchBatchedMemory(real **basePtr, unsigned elementSize, bool clean) {
    real *element = basePtr[blockIdx.x];
    int id = threadIdx.x;
    while (id < elementSize) {
      if (clean) {
        element[id] = 0.0;
      } else {
        real value = element[id];
        // Do something dummy here. We just need to check the pointers point to valid memory locations.
        // Avoid compiler optimization. Possibly, implement a dummy code with asm.
        value += 1.0;
        value -= 1.0;
      }
      id += blockDim.x;
    }
  }

  void Algorithms::touchBatchedMemory(real **basePtr,
                                      unsigned elementSize,
                                      unsigned numElements,
                                      bool clean,
                                      void* streamPtr) {
    dim3 block(256, 1, 1);
    dim3 grid(numElements, 1, 1);
    auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
    kernel_touchBatchedMemory<<<grid, block, 0, stream>>>(basePtr, elementSize, clean); CHECK_ERR;
  }

//--------------------------------------------------------------------------------------------------
__global__  void kernel_setToValue(real** out, real value, size_t elementSize, size_t numElements) {
  const int elementId = blockIdx.x;
  if (elementId < numElements) {
    real *element = out[elementId];
    const int tid = threadIdx.x;
    for (int i = tid; i < elementSize; i += blockDim.x) {
      element[i] = value;
    }
  }
}

void Algorithms::setToValue(real** out, real value, size_t elementSize, size_t numElements, void* streamPtr) {
  dim3 block(256, 1, 1);
  dim3 grid(numElements, 1, 1);
  auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
  kernel_setToValue<<<grid, block, 0, stream>>>(out, value, elementSize, numElements);
  CHECK_ERR;
}

//--------------------------------------------------------------------------------------------------
  template<typename T>
  __global__ void kernel_copyUniformToScatter(T *src, T **dst, size_t srcOffset, size_t copySize) {
    T *srcElement = &src[blockIdx.x * srcOffset];
    T *dstElement = dst[blockIdx.x];
    for (int index = threadIdx.x; index < copySize; index += blockDim.x) {
      dstElement[index] = srcElement[index];
    }
  }

  template<typename T>
  void Algorithms::copyUniformToScatter(T *src,
                                        T **dst,
                                        size_t srcOffset,
                                        size_t copySize,
                                        size_t numElements,
                                        void* streamPtr) {
    dim3 block(256, 1, 1);
    dim3 grid(numElements, 1, 1);
    auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
    kernel_copyUniformToScatter<<<grid, block, 0, stream>>>(src, dst, srcOffset, copySize); CHECK_ERR;
    CHECK_ERR;
  }
  template void Algorithms::copyUniformToScatter(real *src,
                                                 real **dst,
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
  __global__ void kernel_copyScatterToUniform(T **src, T *dst, size_t dstOffset, size_t copySize) {
    T *srcElement = src[blockIdx.x];
    T *dstElement = &dst[blockIdx.x * dstOffset];
    for (int index = threadIdx.x; index < copySize; index += blockDim.x) {
      dstElement[index] = srcElement[index];
    }
  }

  template<typename T>
  void Algorithms::copyScatterToUniform(T **src,
                                        T *dst,
                                        size_t dstOffset,
                                        size_t copySize,
                                        size_t numElements,
                                        void* streamPtr) {
    dim3 block(256, 1, 1);
    dim3 grid(numElements, 1, 1);
    auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
    kernel_copyScatterToUniform<<<grid, block, 0, stream>>>(src, dst, dstOffset, copySize); CHECK_ERR;
    CHECK_ERR;
  }
  template void Algorithms::copyScatterToUniform(real **src,
                                                 real *dst,
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
