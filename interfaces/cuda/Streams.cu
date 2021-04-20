#include "CudaWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"
#include <algorithm>

using namespace device;

void *ConcreteAPI::getNextCircularStream() {
  void *returnStream = static_cast<void *>(m_circularStreamBuffer[m_circularStreamCounter]);
  m_circularStreamCounter += 1;
  if (m_circularStreamCounter >= m_circularStreamBuffer.size()) {
    m_circularStreamCounter = 0;
  }
  return returnStream;
}

void ConcreteAPI::resetCircularStreamCounter() { m_circularStreamCounter = 0; }

size_t ConcreteAPI::getCircularStreamSize() { return m_circularStreamBuffer.size(); }

void ConcreteAPI::syncStreamFromCircularBuffer(void *streamPtr) {
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
#ifndef NDEBUG
  auto itr = std::find(m_circularStreamBuffer.begin(), m_circularStreamBuffer.end(), stream);
  if (itr == m_circularStreamBuffer.end()) {
    logError() << "DEVICE::ERROR: passed stream does not belong to circular stream buffer";
  }
#endif
  cudaStreamSynchronize(stream);
  CHECK_ERR;
}

void ConcreteAPI::syncCircularBuffer() {
  for (auto &stream : m_circularStreamBuffer) {
    cudaStreamSynchronize(stream);
    CHECK_ERR;
  }
}

__global__ void kernel_synchAllStreams() {
  // NOTE: an empty stream. It is supposed to get called with Cuda default stream. It is going to
  // force all other streams to finish their tasks
}

void ConcreteAPI::fastStreamsSync() { kernel_synchAllStreams<<<1, 1>>>(); }

void *ConcreteAPI::getDefaultStream() { return NULL; }