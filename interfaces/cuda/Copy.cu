#include "utils/logger.h"
#include "CudaWrappedAPI.h"
#include "Internals.h"
#include <algorithm>

using namespace device;


void ConcreteAPI::copyTo(void* Dst, const void* Src, size_t Count) {
  cudaMemcpy(Dst, Src, Count, cudaMemcpyHostToDevice); CHECK_ERR;
  m_Statistics.ExplicitlyTransferredDataToDeviceBytes += Count;
}


void ConcreteAPI::copyFrom(void* Dst, const void* Src, size_t Count) {
  cudaMemcpy(Dst, Src, Count, cudaMemcpyDeviceToHost); CHECK_ERR;
  m_Statistics.ExplicitlyTransferredDataToHostBytes += Count;
}


void ConcreteAPI::copyBetween(void* Dst, const void* Src, size_t Count) {
  cudaMemcpy(Dst, Src, Count, cudaMemcpyDeviceToDevice); CHECK_ERR;
}


void ConcreteAPI::copy2dArrayTo(void *Dst,
                                size_t Dpitch,
                                const void *Src,
                                size_t Spitch,
                                size_t Width,
                                size_t Height) {
  cudaMemcpy2D(Dst, Dpitch, Src, Spitch, Width, Height, cudaMemcpyHostToDevice); CHECK_ERR;
  m_Statistics.ExplicitlyTransferredDataToDeviceBytes += Width * Height;
}


void ConcreteAPI::copy2dArrayFrom(void *Dst,
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
                                         unsigned ElementSize) {

  real *SrcElement = BaseSrcPtr[blockIdx.x];
  real *DstElement = BaseDstPtr[blockIdx.x];
  for (int Index = threadIdx.x; Index < ElementSize; Index += blockDim.x) {
    DstElement[Index] = SrcElement[Index];
  }
}

void ConcreteAPI::streamBatchedData(real **BaseSrcPtr,
                                    real **BaseDstPtr,
                                    unsigned ElementSize,
                                    unsigned NumElements) {
  dim3 Block(internals::WARP_SIZE, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_streamBatchedData<<<Grid, Block>>>(BaseSrcPtr, BaseDstPtr, ElementSize); CHECK_ERR;
}

__global__ void kernel_accumulateBatchedData(real **BaseSrcPtr,
                                             real **BaseDstPtr,
                                             unsigned ElementSize) {

  real *SrcElement = BaseSrcPtr[blockIdx.x];
  real *DstElement = BaseDstPtr[blockIdx.x];
  for (int Index = threadIdx.x; Index < ElementSize; Index += blockDim.x) {
    DstElement[Index] += SrcElement[Index];
  }
}

void ConcreteAPI::accumulateBatchedData(real **BaseSrcPtr,
                                        real **BaseDstPtr,
                                        unsigned ElementSize,
                                        unsigned NumElements) {
  dim3 Block(internals::WARP_SIZE, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_accumulateBatchedData<<<Grid, Block>>>(BaseSrcPtr, BaseDstPtr, ElementSize); CHECK_ERR;
}


void ConcreteAPI::prefetchUnifiedMemTo(Destination Type, const void* DevPtr, size_t Count, void* streamPtr) {
  cudaStream_t stream = (streamPtr == nullptr) ? 0 : *(static_cast<cudaStream_t*>(streamPtr));
#ifndef NDEBUG
  auto itr = std::find(m_circularStreamBuffer.begin(), m_circularStreamBuffer.end(), stream);
  if (itr == m_circularStreamBuffer.end()) {
    logError() << "DEVICE::ERROR: passed stream does not belong to circular stream buffer";
  }
#endif
  cudaMemPrefetchAsync(DevPtr,
                       Count,
                       Type == Destination::CurrentDevice ? m_CurrentDeviceId : cudaCpuDeviceId,
                       stream); CHECK_ERR;
}
