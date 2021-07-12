#include "CudaWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"
#include <algorithm>
#include <cassert>

using namespace device;


void *ConcreteAPI::getDefaultStream() {
  return m_defaultStream;
 }

void ConcreteAPI::syncDefaultStreamWithHost() {
  cudaStreamSynchronize(m_defaultStream);
  CHECK_ERR;
}

void *ConcreteAPI::getNextCircularStream() {
  assert(m_isCircularStreamsForked && "use a circular stream must be used inside a forked region");
  void *returnStream = static_cast<void *>(m_circularStreamBuffer[m_circularStreamCounter]);
  m_circularStreamCounter += 1;
  if (m_circularStreamCounter >= m_circularStreamBuffer.size()) {
    m_circularStreamCounter = 0;
  }
  return returnStream;
  //return m_defaultStream;
}

void ConcreteAPI::resetCircularStreamCounter() {
  m_circularStreamCounter = 0;
}

size_t ConcreteAPI::getCircularStreamSize() { return m_circularStreamBuffer.size(); }

void ConcreteAPI::syncStreamFromCircularBufferWithHost(void *userStream) {
  cudaStream_t stream = static_cast<cudaStream_t>(userStream);
#ifndef NDEBUG
  auto itr = std::find(m_circularStreamBuffer.begin(), m_circularStreamBuffer.end(), stream);
  if (itr == m_circularStreamBuffer.end()) {
    logError() << "DEVICE::ERROR: passed stream does not belong to circular stream buffer";
  }
#endif
  cudaStreamSynchronize(stream);
  CHECK_ERR;
}

void ConcreteAPI::syncCircularBuffersWithHost() {
  for (auto &stream : m_circularStreamBuffer) {
    cudaStreamSynchronize(stream);
    CHECK_ERR;
  }
}


void ConcreteAPI::forkCircularStreamsFromDefault() {
  assert(!m_isCircularStreamsForked && "circular streams must be joined before forking");

  cudaEventRecord(m_defaultStreamEvent, m_defaultStream);
  CHECK_ERR;

  for (auto &stream : m_circularStreamBuffer) {
    cudaStreamWaitEvent(stream, m_defaultStreamEvent, 0);
    CHECK_ERR;
  }
  m_isCircularStreamsForked = true;
}


void ConcreteAPI::joinCircularStreamsToDefault() {
  assert(m_isCircularStreamsForked && "circular streams must be forked before joining");

  for (size_t i = 0; i < m_circularStreamBuffer.size(); ++i) {
    cudaEventRecord(m_circularStreamEvents[i], m_circularStreamBuffer[i]);
    CHECK_ERR;
  }

  for (size_t i = 0; i < m_circularStreamBuffer.size(); ++i) {
    cudaStreamWaitEvent(m_defaultStream, m_circularStreamEvents[i], 0);
    CHECK_ERR;
  }
  m_isCircularStreamsForked = false;
}


bool ConcreteAPI::isCircularStreamsJoinedWithDefault() {
  return !m_isCircularStreamsForked;
}