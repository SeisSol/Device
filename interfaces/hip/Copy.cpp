#include "utils/logger.h"
#include "hip/hip_runtime.h"
#include "HipWrappedAPI.h"
#include "Internals.h"
#include <algorithm>

using namespace device;

void ConcreteAPI::copyTo(void* Dst, const void* Src, size_t Count) {
  hipMemcpy(Dst, Src, Count, hipMemcpyHostToDevice); CHECK_ERR;
  m_Statistics.ExplicitlyTransferredDataToDeviceBytes += Count;
}


void ConcreteAPI::copyFrom(void* Dst, const void* Src, size_t Count) {
  hipMemcpy(Dst, Src, Count, hipMemcpyDeviceToHost); CHECK_ERR;
  m_Statistics.ExplicitlyTransferredDataToHostBytes += Count;
}


void ConcreteAPI::copyBetween(void* Dst, const void* Src, size_t Count) {
  hipMemcpy(Dst, Src, Count, hipMemcpyDeviceToDevice); CHECK_ERR;
}


void ConcreteAPI::copy2dArrayTo(void *Dst,
                                size_t Dpitch,
                                const void *Src,
                                size_t Spitch,
                                size_t Width,
                                size_t Height) {
  hipMemcpy2D(Dst, Dpitch, Src, Spitch, Width, Height, hipMemcpyHostToDevice); CHECK_ERR;
  m_Statistics.ExplicitlyTransferredDataToDeviceBytes += Width * Height;
}


void ConcreteAPI::copy2dArrayFrom(void *Dst,
                                  size_t Dpitch,
                                  const void *Src,
                                  size_t Spitch,
                                  size_t Width,
                                  size_t Height) {
  hipMemcpy2D(Dst, Dpitch, Src, Spitch, Width, Height, hipMemcpyDeviceToHost); CHECK_ERR;
  m_Statistics.ExplicitlyTransferredDataToHostBytes += Width * Height;
}


void ConcreteAPI::prefetchUnifiedMemTo(Destination Type, const void* DevPtr, size_t Count, void* streamPtr) {
#ifndef NDEBUG
  auto itr = std::find(m_circularStreamBuffer.begin(), m_circularStreamBuffer.end(), stream);
  if (itr == m_circularStreamBuffer.end()) {
    logError() << "DEVICE::ERROR: passed stream does not belong to circular stream buffer";
  }
#endif
  //cudaMemPrefetchAsync - Not supported by HIP
  if(Type == Destination::CurrentDevice){
    hipMemcpyAsync((void*) DevPtr, DevPtr, Count, hipMemcpyHostToDevice, Stream);
  }else{
    hipMemcpyAsync((void*) DevPtr, DevPtr, Count, hipMemcpyDeviceToHost, Stream);
  }
}
