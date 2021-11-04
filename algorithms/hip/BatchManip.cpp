#include "AbstractAPI.h"
#include "interfaces/hip/Internals.h"
#include <cassert>
#include <device.h>

namespace device {
__global__ void kernel_streamBatchedData(real **baseSrcPtr, 
                                         real **baseDstPtr, 
                                         unsigned elementSize) {

  real *srcElement = baseSrcPtr[hipBlockIdx_x];
  real *dstElement = baseDstPtr[hipBlockIdx_x];
  for (int index = hipThreadIdx_x; index < elementSize; index += hipBlockDim_x) {
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
  hipLaunchKernelGGL(kernel_streamBatchedData, grid, block, 0, stream, baseSrcPtr, baseDstPtr, elementSize);
  CHECK_ERR;
}

//--------------------------------------------------------------------------------------------------
__global__ void kernel_accumulateBatchedData(real **baseSrcPtr, 
                                             real **baseDstPtr, 
                                             unsigned elementSize) {

  real *srcElement = baseSrcPtr[hipBlockIdx_x];
  real *dstElement = baseDstPtr[hipBlockIdx_x];
  for (int index = hipThreadIdx_x; index < elementSize; index += hipBlockDim_x) {
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
  hipLaunchKernelGGL(kernel_accumulateBatchedData, grid, block, 0, stream, baseSrcPtr, baseDstPtr, elementSize);
  CHECK_ERR;
}

//--------------------------------------------------------------------------------------------------
__global__ void kernel_touchBatchedMemory(real **basePtr, 
                                          unsigned elementSize, 
                                          bool clean) {
  real *element = basePtr[hipBlockIdx_x];
  int id = hipThreadIdx_x;
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
    id += hipBlockDim_x;
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
  hipLaunchKernelGGL(kernel_touchBatchedMemory, grid, block, 0, stream, basePtr, elementSize, clean);
  CHECK_ERR;
}

//--------------------------------------------------------------------------------------------------
template<typename T>
__global__ void kernel_copyUniformToScatter(T *src, T **dst, size_t chunkSize) {
  T *srcElement = &src[hipBlockIdx_x];
  T *dstElement = dst[hipBlockIdx_x];
  for (int index = hipThreadIdx_x; index < chunkSize; index += blockDim.x) {
    dstElement[index] = srcElement[index];
  }
}

template<typename T>
void Algorithms::copyUniformToScatter(T *src, 
                                      T **dst, 
                                      size_t chunkSize, 
                                      size_t numElements, 
                                      void* streamPtr) {
  dim3 block(256, 1, 1);
  dim3 grid(numElements, 1, 1);
  auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
  hipLaunchKernelGGL(kernel_copyUniformToScatter, grid, block, 0, stream, src, dst, chunkSize);
  CHECK_ERR;
}
template void Algorithms::copyUniformToScatter(real *src, real **dst, size_t chunkSize, size_t numElements, void* streamPtr);
template void Algorithms::copyUniformToScatter(int *src, int **dst, size_t chunkSize, size_t numElements, void* streamPtr);
template void Algorithms::copyUniformToScatter(char *src, char **dst, size_t chunkSize, size_t numElements, void* streamPtr);

//--------------------------------------------------------------------------------------------------
template<typename T>
__global__ void kernel_copyScatterToUniform(T **src, T *dst, size_t chunkSize) {
  T *srcElement = src[hipBlockIdx_x];
  T *dstElement = &dst[hipBlockIdx_x];
  for (int index = hipThreadIdx_x; index < chunkSize; index += blockDim.x) {
    dstElement[index] = srcElement[index];
  }
}

template<typename T>
void Algorithms::copyScatterToUniform(T **src, 
                                      T *dst, 
                                      size_t chunkSize, 
                                      size_t numElements, 
                                      void* streamPtr) {
  dim3 block(256, 1, 1);
  dim3 grid(numElements, 1, 1);
  auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);
  hipLaunchKernelGGL(kernel_copyScatterToUniform, grid, block, 0, stream, src, dst, chunkSize);
  CHECK_ERR;
}
template void Algorithms::copyScatterToUniform(real **src, real *dst, size_t chunkSize, size_t numElements, void* streamPtr);
template void Algorithms::copyScatterToUniform(int **src, int *dst, size_t chunkSize, size_t numElements, void* streamPtr);
template void Algorithms::copyScatterToUniform(char **src, char *dst, size_t chunkSize, size_t numElements, void* streamPtr);
} // namespace device
