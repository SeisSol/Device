#include "CudaWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"
#include <algorithm>

using namespace device;

void ConcreteAPI::copyTo(void *dst, const void *src, size_t count) {
  cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
  CHECK_ERR;
  m_statistics.explicitlyTransferredDataToDeviceBytes += count;
}

void ConcreteAPI::copyFrom(void *dst, const void *src, size_t count) {
  cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
  CHECK_ERR;
  m_statistics.explicitlyTransferredDataToHostBytes += count;
}

void ConcreteAPI::copyBetween(void *dst, const void *src, size_t count) {
  cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
  CHECK_ERR;
}

void ConcreteAPI::copy2dArrayTo(void *dst, size_t dpitch, const void *src, size_t spitch,
                                size_t width, size_t height) {
  cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice);
  CHECK_ERR;
  m_statistics.explicitlyTransferredDataToDeviceBytes += width * height;
}

void ConcreteAPI::copy2dArrayFrom(void *dst, size_t dpitch, const void *src, size_t spitch,
                                  size_t width, size_t height) {
  cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost);
  CHECK_ERR;
  m_statistics.explicitlyTransferredDataToHostBytes += width * height;
}

void ConcreteAPI::prefetchUnifiedMemTo(Destination type, const void *devPtr, size_t count,
                                       void *streamPtr) {
  cudaStream_t stream = (streamPtr == nullptr) ? 0 : (static_cast<cudaStream_t>(streamPtr));
#ifndef NDEBUG
  auto itr = std::find(m_circularStreamBuffer.begin(), m_circularStreamBuffer.end(), stream);
  if (itr == m_circularStreamBuffer.end()) {
    logError() << "DEVICE::ERROR: passed stream does not belong to circular stream buffer";
  }
#endif
  cudaMemPrefetchAsync(devPtr,
                       count,
                       type == Destination::CurrentDevice ? m_currentDeviceId : cudaCpuDeviceId,
                       stream);
  CHECK_ERR;
}
