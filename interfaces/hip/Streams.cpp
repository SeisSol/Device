#include "utils/logger.h"
#include "HipWrappedAPI.h"
#include "Internals.h"
#include <algorithm>

using namespace device;

void *ConcreteAPI::getNextCircularStream() {
  isFlagSet<InterfaceInitialized>(status);
  void* returnStream = static_cast<void*>(circularStreamBuffer[circularStreamCounter]);
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

void ConcreteAPI::syncStreamFromCircularBuffer(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
#ifndef NDEBUG
  auto itr = std::find(circularStreamBuffer.begin(), circularStreamBuffer.end(), stream);
  if (itr == circularStreamBuffer.end()) {
    logError() << "DEVICE::ERROR: passed stream does not belong to circular stream buffer";
  }
#endif
  hipStreamSynchronize(stream); CHECK_ERR;
}

void ConcreteAPI::syncCircularBuffer() {
  isFlagSet<InterfaceInitialized>(status);
  for (auto& stream: circularStreamBuffer) {
    hipStreamSynchronize(stream); CHECK_ERR;
  }
}

__global__ void kernel_synchAllStreams() {
  // NOTE: an empty stream. It is supposed to get called with Cuda default stream. It is going to force all
  // other streams to finish their tasks
}

void ConcreteAPI::fastStreamsSync() {
  isFlagSet<DeviceSelected>(status);
  hipLaunchKernelGGL(kernel_synchAllStreams, dim3(1), dim3(1), 0, 0);
}

void *ConcreteAPI::getDefaultStream() {
  isFlagSet<InterfaceInitialized>(status);
  return static_cast<void *>(defaultStream);
}