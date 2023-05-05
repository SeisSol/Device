#include "CudaWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"
#include <algorithm>
#include <cassert>
#include <sstream>


using namespace device;

void* ConcreteAPI::getDefaultStream() {
  isFlagSet<InterfaceInitialized>(status);
  return static_cast<void *>(defaultStream);
}

void ConcreteAPI::syncDefaultStreamWithHost() {
  isFlagSet<InterfaceInitialized>(status);
  cudaStreamSynchronize(defaultStream);
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
  isFlagSet<CircularStreamBufferInitialized>(status);

  for (auto &stream : circularStreamBuffer) {
    cudaStreamSynchronize(stream);
    CHECK_ERR;
  }
}


void ConcreteAPI::forkCircularStreamsFromDefault() {
  isFlagSet<CircularStreamBufferInitialized>(status);
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
  isFlagSet<CircularStreamBufferInitialized>(status);
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
  isFlagSet<CircularStreamBufferInitialized>(status);
  return !isCircularStreamsForked;
}


void* ConcreteAPI::createGenericStream() {
  isFlagSet<InterfaceInitialized>(status);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); CHECK_ERR;
  genericStreams.insert(stream);
  return reinterpret_cast<void*>(stream);
}


void ConcreteAPI::destroyGenericStream(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
  auto it = genericStreams.find(stream);
  if (it != genericStreams.end()) {
    genericStreams.erase(it);
  }
  cudaStreamDestroy(stream);
  CHECK_ERR;
}


void ConcreteAPI::syncStreamWithHost(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
  cudaStreamSynchronize(stream);
  CHECK_ERR;
}


bool ConcreteAPI::isStreamWorkDone(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
  auto streamStatus = cudaStreamQuery(stream);

  if (streamStatus == cudaSuccess) {
    return true;
  }
  else {
    // dump the last error e.g., cudaErrorInvalidResourceHandle
    cudaGetLastError();
    return false;
  }
}
