#include "utils/logger.h"
#include "HipWrappedAPI.h"
#include "Internals.h"
#include <algorithm>

using namespace device;

void *ConcreteAPI::getDefaultStream() {
  return m_defaultStream;
}

void ConcreteAPI::syncDefaultStreamWithHost() {
  hipStreamSynchronize(m_defaultStream);
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
}

void ConcreteAPI::resetCircularStreamCounter() {
  m_circularStreamCounter = 0;
}

size_t ConcreteAPI::getCircularStreamSize() {
  return m_circularStreamBuffer.size();
}

void ConcreteAPI::syncStreamFromCircularBufferWithHost(void *userStream) {
  hipStream_t stream = static_cast<hipStream_t>(userStream);
#ifndef NDEBUG
  auto itr = std::find(m_circularStreamBuffer.begin(), m_circularStreamBuffer.end(), stream);
  if (itr == m_circularStreamBuffer.end()) {
    logError() << "DEVICE::ERROR: passed stream does not belong to circular stream buffer";
  }
#endif
  hipStreamSynchronize(stream);
  CHECK_ERR;
}

void ConcreteAPI::syncCircularBuffersWithHost() {
  for (auto &stream : m_circularStreamBuffer) {
    hipStreamSynchronize(stream);
    CHECK_ERR;
  }
}

void ConcreteAPI::forkCircularStreamsFromDefault() {
  assert(!m_isCircularStreamsForked && "circular streams must be joined before forking");

  hipEventRecord(m_defaultStreamEvent, m_defaultStream);
  CHECK_ERR;

  for (auto &stream : m_circularStreamBuffer) {
    hipStreamWaitEvent(stream, m_defaultStreamEvent, 0);
    CHECK_ERR;
  }
  m_isCircularStreamsForked = true;
}


void ConcreteAPI::joinCircularStreamsToDefault() {
  assert(m_isCircularStreamsForked && "circular streams must be forked before joining");

  for (size_t i = 0; i < m_circularStreamBuffer.size(); ++i) {
    hipEventRecord(m_circularStreamEvents[i], m_circularStreamBuffer[i]);
    CHECK_ERR;
  }

  for (size_t i = 0; i < m_circularStreamBuffer.size(); ++i) {
    hipStreamWaitEvent(m_defaultStream, m_circularStreamEvents[i], 0);
    CHECK_ERR;
  }
  m_isCircularStreamsForked = false;
}

bool ConcreteAPI::isCircularStreamsJoinedWithDefault() {
  return !m_isCircularStreamsForked;
}
