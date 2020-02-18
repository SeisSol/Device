#include <assert.h>

#include "CudaInterface.h"
#include "Internals.h"

using namespace device;


void ConcreteInterface::copyTo(void* Dst, const void* Src, size_t Count) {
  cudaMemcpy(Dst, Src, Count, cudaMemcpyHostToDevice); CHECK_ERR;
  m_Statistics.ExplicitlyTransferredDataToDeviceBytes += Count;
}


void ConcreteInterface::copyFrom(void* Dst, const void* Src, size_t Count) {
  cudaMemcpy(Dst, Src, Count, cudaMemcpyDeviceToHost); CHECK_ERR;
  m_Statistics.ExplicitlyTransferredDataToHostBytes += Count;
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
  m_Statistics.ExplicitlyTransferredDataToDeviceBytes += Width * Height;
}


void ConcreteInterface::copy2dArrayFrom(void *Dst,
                                        size_t Dpitch,
                                        const void *Src,
                                        size_t Spitch,
                                        size_t Width,
                                        size_t Height) {
  cudaMemcpy2D(Dst, Dpitch, Src, Spitch, Width, Height, cudaMemcpyDeviceToHost); CHECK_ERR;
  m_Statistics.ExplicitlyTransferredDataToHostBytes += Width * Height;
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


void ConcreteInterface::prefetchUnifiedMemTo(Destination Type, const void* DevPtr, size_t Count, int StreamId) {
  cudaStream_t Stream = StreamId == 0 ? 0 : *m_IdToStreamMap[StreamId];
#ifndef NDEBUG
  if (Stream != 0) {
    assert((m_IdToStreamMap.find(StreamId) != m_IdToStreamMap.end())
           && "DEVICE: stream doesn't exist. cannot prefetch memory");
  }
#endif
  cudaMemPrefetchAsync(DevPtr,
                       Count,
                       Type == Destination::CurrentDevice ? m_CurrentDeviceId : cudaCpuDeviceId,
                       Stream); CHECK_ERR;
}
