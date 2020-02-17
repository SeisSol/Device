#include "CudaInterface.h"
#include "Internals.h"

using namespace device;


void ConcreteInterface::copyTo(void* Dst, const void* Src, size_t Count) {
  cudaMemcpy(Dst, Src, Count, cudaMemcpyHostToDevice); CHECK_ERR;
}


void ConcreteInterface::copyFrom(void* Dst, const void* Src, size_t Count) {
  cudaMemcpy(Dst, Src, Count, cudaMemcpyDeviceToHost); CHECK_ERR;
}


void ConcreteInterface::copyBetween(void* Dst, const void* Src, size_t Count) {
  cudaMemcpy(Dst, Src, Count, cudaMemcpyDeviceToDevice); CHECK_ERR;
}


void ConcreteInterface::copy2dArrayTo(void *Dst,
                                      size_t Dpitch,
                                      const void *Src,
                                      size_t Spitch,
                                      size_t Width,
                                      size_t Height) {
  cudaMemcpy2D(Dst, Dpitch, Src, Spitch, Width, Height, cudaMemcpyHostToDevice); CHECK_ERR;
}


void ConcreteInterface::copy2dArrayFrom(void *Dst,
                                        size_t Dpitch,
                                        const void *Src,
                                        size_t Spitch,
                                        size_t Width,
                                        size_t Height) {
  cudaMemcpy2D(Dst, Dpitch, Src, Spitch, Width, Height, cudaMemcpyDeviceToHost); CHECK_ERR;
}


__global__ void kernel_streamBatchedData(real **BaseSrcPtr,
                                         real **BaseDstPtr,
                                         unsigned ElementSize,
                                         bool Accumulate) {

  real *SrcElement = BaseSrcPtr[blockIdx.x];
  real *DstElement = BaseDstPtr[blockIdx.x];

  int Id = threadIdx.x;
  while (Id < ElementSize) {
    if (Accumulate) {
      DstElement[Id] += SrcElement[Id];
    } else {
      DstElement[Id] = SrcElement[Id];
    }
    Id += blockDim.x;
  }
}

void ConcreteInterface::streamBatchedData(real **BaseSrcPtr,
                                          real **BaseDstPtr,
                                          unsigned ElementSize,
                                          unsigned NumElements) {
  dim3 Block(256, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_streamBatchedData<<<Grid, Block>>>(BaseSrcPtr, BaseDstPtr, ElementSize, false); CHECK_ERR;
}


void ConcreteInterface::accumulateBatchedData(real **BaseSrcPtr,
                                              real **BaseDstPtr,
                                              unsigned ElementSize,
                                              unsigned NumElements) {
  dim3 Block(ElementSize, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_streamBatchedData<<<Grid, Block>>>(BaseSrcPtr, BaseDstPtr, ElementSize, true); CHECK_ERR;
}