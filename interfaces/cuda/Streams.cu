#include "CudaWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"
#include <algorithm>
#include <cassert>

using namespace device;


void *ConcreteAPI::getDefaultStream() {
  return defaultStream;
}

void ConcreteAPI::syncDefaultStreamWithHost() {
  cudaStreamSynchronize(defaultStream);
  CHECK_ERR;
}

void *ConcreteAPI::getNextCircularStream() {
  assert(isCircularStreamsForked && "use a circular stream must be used inside a forked region");
  void *returnStream = static_cast<void *>(circularStreamBuffer[circularStreamCounter]);
  circularStreamCounter += 1;
  if (circularStreamCounter >= circularStreamBuffer.size()) {
    circularStreamCounter = 0;
  }
  return returnStream;
}

void ConcreteAPI::resetCircularStreamCounter() {
  circularStreamCounter = 0;
}

size_t ConcreteAPI::getCircularStreamSize() { return circularStreamBuffer.size(); }

void ConcreteAPI::syncStreamFromCircularBufferWithHost(void *userStream) {
  cudaStream_t stream = static_cast<cudaStream_t>(userStream);
#ifndef NDEBUG
  auto itr = std::find(circularStreamBuffer.begin(), circularStreamBuffer.end(), stream);
  if (itr == circularStreamBuffer.end()) {
    logError() << "DEVICE::ERROR: passed stream does not belong to circular stream buffer";
  }
#endif
  cudaStreamSynchronize(stream);
  CHECK_ERR;
}

void ConcreteAPI::syncCircularBuffersWithHost() {
  for (auto &stream : circularStreamBuffer) {
    cudaStreamSynchronize(stream);
    CHECK_ERR;
  }
}


void ConcreteAPI::forkCircularStreamsFromDefault() {
  assert(!isCircularStreamsForked && "circular streams must be joined before forking");

  cudaEventRecord(defaultStreamEvent, defaultStream);
  CHECK_ERR;

  for (auto &stream : circularStreamBuffer) {
    cudaStreamWaitEvent(stream, defaultStreamEvent, 0);
    CHECK_ERR;
  }
  isCircularStreamsForked = true;
}


void ConcreteAPI::joinCircularStreamsToDefault() {
  assert(isCircularStreamsForked && "circular streams must be forked before joining");

  for (size_t i = 0; i < circularStreamBuffer.size(); ++i) {
    cudaEventRecord(circularStreamEvents[i], circularStreamBuffer[i]);
    CHECK_ERR;
  }

  for (size_t i = 0; i < circularStreamBuffer.size(); ++i) {
    cudaStreamWaitEvent(defaultStream, circularStreamEvents[i], 0);
    CHECK_ERR;
  }
  isCircularStreamsForked = false;
}


bool ConcreteAPI::isCircularStreamsJoinedWithDefault() {
  return !isCircularStreamsForked;
}
