#include "HipWrappedAPI.h"
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
  hipStreamSynchronize(defaultStream);
  CHECK_ERR;
}

void* ConcreteAPI::createStream(double priority) {
  isFlagSet<InterfaceInitialized>(status);
  hipStream_t stream;
  const auto truePriority = mapPercentage(priorityMin, priorityMax, priority);
  hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority); CHECK_ERR;
  genericStreams.insert(stream);
  return reinterpret_cast<void*>(stream);
}


void ConcreteAPI::destroyGenericStream(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
  auto it = genericStreams.find(stream);
  if (it != genericStreams.end()) {
    genericStreams.erase(it);
  }
  hipStreamDestroy(stream);
  CHECK_ERR;
}


void ConcreteAPI::syncStreamWithHost(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
  hipStreamSynchronize(stream);
  CHECK_ERR;
}


bool ConcreteAPI::isStreamWorkDone(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
  auto streamStatus = hipStreamQuery(stream);

  if (streamStatus == hipSuccess) {
    return true;
  }
  else {
    // dump the last error e.g., hipErrorInvalidResourceHandle
    hipGetLastError();
    return false;
  }
}

void ConcreteAPI::syncStreamWithEvent(void* streamPtr, void* eventPtr) {
  isFlagSet<InterfaceInitialized>(status);
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  hipStreamWaitEvent(stream, event, 0);
  CHECK_ERR;
}

namespace {
static void streamCallback(hipStream_t, hipError_t, void* data) {
  auto* function = reinterpret_cast<std::function<void()>*>(data);
  (*function)();
  delete function;
}
} // namespace

void ConcreteAPI::streamHostFunction(void* streamPtr, const std::function<void()>& function) {
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
  auto* functionData = new std::function<void()>(function);
  hipStreamAddCallback(stream, streamCallback, functionData, 0);
  CHECK_ERR;
}

namespace {
__global__ void spinloop(uint32_t* location, uint32_t value) {
  volatile uint32_t* spinLocation = location;
  while (true) {
    if (*spinLocation >= value) {
      return;
    }
    __threadfence_system();
  }
}
}

void ConcreteAPI::streamWaitMemory(void* streamPtr, uint32_t* location, uint32_t value) {
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
  uint32_t* deviceLocation = nullptr;
  hipHostGetDevicePointer(reinterpret_cast<void**>(&deviceLocation), location, 0);
  CHECK_ERR;
  const auto result = hipStreamWaitValue32(stream, deviceLocation, value, hipStreamWaitValueGte, 0xffffffff);
  if (result == hipErrorNotSupported) {
    spinloop<<<1,1,0,stream>>>(deviceLocation, value);
  }
  CHECK_ERR;
}
