#include "HipWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"
#include <algorithm>
#include <cassert>

using namespace device;

void* ConcreteAPI::getDefaultStream() {
  isFlagSet<InterfaceInitialized>(status);
  return static_cast<void *>(defaultStream);
}

void ConcreteAPI::syncDefaultStreamWithHost() {
  isFlagSet<InterfaceInitialized>(status);
  hipStreamSynchronize(defaultStream);
  CHECK_ERR;
}

void* ConcreteAPI::getNextCircularStream() {
  isFlagSet<CircularStreamBufferInitialized>(status);
  assert(isCircularStreamsForked && "use a circular stream must be used inside a forked region");
  auto* returnStream = static_cast<void *>(circularStreamBuffer[circularStreamCounter]);
  circularStreamCounter += 1;
  if (circularStreamCounter >= circularStreamBuffer.size()) {
    circularStreamCounter = 0;
  }
  return returnStream;
}

void ConcreteAPI::resetCircularStreamCounter() {
  isFlagSet<CircularStreamBufferInitialized>(status);
  circularStreamCounter = 0;
}

size_t ConcreteAPI::getCircularStreamSize() {
  isFlagSet<CircularStreamBufferInitialized>(status);
  return circularStreamBuffer.size();
}

void ConcreteAPI::syncStreamFromCircularBufferWithHost(void* userStream) {
  isFlagSet<CircularStreamBufferInitialized>(status);

  hipStream_t stream = static_cast<hipStream_t>(userStream);
#ifndef NDEBUG
  auto itr = std::find(circularStreamBuffer.begin(), circularStreamBuffer.end(), stream);
  if (itr == circularStreamBuffer.end()) {
    logError() << "DEVICE::ERROR: passed stream does not belong to circular stream buffer";
  }
#endif
  hipStreamSynchronize(stream);
  CHECK_ERR;
}

void ConcreteAPI::syncCircularBuffersWithHost() {
  isFlagSet<CircularStreamBufferInitialized>(status);

  for (auto &stream : circularStreamBuffer) {
    hipStreamSynchronize(stream);
    CHECK_ERR;
  }
}


void ConcreteAPI::forkCircularStreamsFromDefault() {
  isFlagSet<CircularStreamBufferInitialized>(status);
  assert(!isCircularStreamsForked && "circular streams must be joined before forking");

  hipEventRecord(defaultStreamEvent, defaultStream);
  CHECK_ERR;

  for (auto &stream : circularStreamBuffer) {
    hipStreamWaitEvent(stream, defaultStreamEvent, 0);
    CHECK_ERR;
  }
  isCircularStreamsForked = true;
}


void ConcreteAPI::joinCircularStreamsToDefault() {
  isFlagSet<CircularStreamBufferInitialized>(status);
  assert(isCircularStreamsForked && "circular streams must be forked before joining");

  for (size_t i = 0; i < circularStreamBuffer.size(); ++i) {
    hipEventRecord(circularStreamEvents[i], circularStreamBuffer[i]);
    CHECK_ERR;
  }

  for (size_t i = 0; i < circularStreamBuffer.size(); ++i) {
    hipStreamWaitEvent(defaultStream, circularStreamEvents[i], 0);
    CHECK_ERR;
  }
  isCircularStreamsForked = false;
}


bool ConcreteAPI::isCircularStreamsJoinedWithDefault() {
  isFlagSet<CircularStreamBufferInitialized>(status);
  return !isCircularStreamsForked;
}
