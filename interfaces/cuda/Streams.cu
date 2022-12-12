#include "CudaWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"
#include <algorithm>

using namespace device;

void *ConcreteAPI::getNextCircularStream() {
  isFlagSet<InterfaceInitialized>(status);
  void *returnStream = static_cast<void *>(circularStreamBuffer[circularStreamCounter]);
  circularStreamCounter += 1;
  if (circularStreamCounter >= circularStreamBuffer.size()) {
    circularStreamCounter = 0;
  }
  return returnStream;
}

void ConcreteAPI::resetCircularStreamCounter() {
  isFlagSet<InterfaceInitialized>(status);
  circularStreamCounter = 0;
}

size_t ConcreteAPI::getCircularStreamSize() {
  isFlagSet<InterfaceInitialized>(status);
  return circularStreamBuffer.size();
}

bool ConcreteAPI::isStreamReady(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
  auto streamStatus = cudaStreamQuery(stream);

  switch (streamStatus) {
    case cudaSuccess: return true;
    case cudaErrorNotReady: return false;
    default: {
      CHECK_ERR;
    }
    return false;
  }
}

void ConcreteAPI::syncStreamFromCircularBuffer(void *streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
#ifndef NDEBUG
  auto itr = std::find(circularStreamBuffer.begin(), circularStreamBuffer.end(), stream);
  if (itr == circularStreamBuffer.end()) {
    logError() << "DEVICE::ERROR: passed stream does not belong to circular stream buffer";
  }
#endif
  cudaStreamSynchronize(stream);
  CHECK_ERR;
}

void ConcreteAPI::syncCircularBuffer() {
  isFlagSet<InterfaceInitialized>(status);
  for (auto &stream : circularStreamBuffer) {
    cudaStreamSynchronize(stream);
    CHECK_ERR;
  }
}

__global__ void kernel_synchAllStreams() {
  // NOTE: an empty stream. It is supposed to get called with Cuda default stream. It is going to
  // force all other streams to finish their tasks
}

void ConcreteAPI::fastStreamsSync() {
  isFlagSet<DeviceSelected>(status);
  kernel_synchAllStreams<<<1, 1>>>();
}

void *ConcreteAPI::getDefaultStream() {
  isFlagSet<InterfaceInitialized>(status);
  return static_cast<void *>(defaultStream);
}