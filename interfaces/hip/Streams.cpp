#include "utils/logger.h"
#include "HipWrappedAPI.h"
#include "Internals.h"
#include <algorithm>

using namespace device;

void *ConcreteAPI::getNextCircularStream() {
  void* returnStream = static_cast<void*>(m_circularStreamBuffer[m_circularStreamCounter]);
  m_circularStreamCounter += 1;
  if (m_circularStreamCounter >= m_circularStreamBuffer.size()) {
    m_circularStreamCounter = 0;
  }
  return returnStream;
}

void ConcreteAPI::resetCircularStreamCounter() {
  m_circularStreamCounter = 0;
}

size_t ConcreteAPI::getCircularStreamSize() {
  return m_circularStreamBuffer.size();
}

void ConcreteAPI::syncStreamFromCircularBuffer(void* streamPtr) {
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
#ifndef NDEBUG
  auto itr = std::find(m_circularStreamBuffer.begin(), m_circularStreamBuffer.end(), stream);
  if (itr == m_circularStreamBuffer.end()) {
    logError() << "DEVICE::ERROR: passed stream does not belong to circular stream buffer";
  }
#endif
  hipStreamSynchronize(stream); CHECK_ERR;
}

void ConcreteAPI::syncCircularBuffer() {
  for (auto& stream: m_circularStreamBuffer) {
    hipStreamSynchronize(stream); CHECK_ERR;
  }
}

__global__ void kernel_synchAllStreams() {
  // NOTE: an empty stream. It is supposed to get called with Cuda default stream. It is going to force all
  // other streams to finish their tasks
}

void ConcreteAPI::fastStreamsSync() {
  hipLaunchKernelGGL(kernel_synchAllStreams, dim3(1), dim3(1), 0, 0);
}
