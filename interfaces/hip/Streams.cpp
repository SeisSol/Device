// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "HipWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"

#include <algorithm>
#include <cassert>
#include <sstream>

using namespace device;

void* ConcreteAPI::getDefaultStream() {
  isFlagSet<InterfaceInitialized>(status);
  return static_cast<void*>(defaultStream);
}

void ConcreteAPI::syncDefaultStreamWithHost() {
  isFlagSet<InterfaceInitialized>(status);

  CHECK_ERR;

  APIWRAP(hipStreamSynchronize(defaultStream));
}

void* ConcreteAPI::createStream(double priority) {
  isFlagSet<InterfaceInitialized>(status);
  hipStream_t stream;
  const auto truePriority = mapPercentage(priorityMin, priorityMax, priority);
  APIWRAP(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority));
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
  APIWRAP(hipStreamDestroy(stream));
}

void ConcreteAPI::syncStreamWithHost(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);

  CHECK_ERR;

  APIWRAP(hipStreamSynchronize(stream));
}

bool ConcreteAPI::isStreamWorkDone(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
  auto streamStatus = APIWRAPX(hipStreamQuery(stream), {hipErrorNotReady});

  return streamStatus == hipSuccess;
}

void ConcreteAPI::syncStreamWithEvent(void* streamPtr, void* eventPtr) {
  isFlagSet<InterfaceInitialized>(status);
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  APIWRAP(hipStreamWaitEvent(stream, event, 0));
}

namespace {
void streamCallbackEpheremal(void* data) {
  auto* function = reinterpret_cast<std::function<void()>*>(data);
  (*function)();
  delete function;
}

void streamCallbackPermanent(void* data) {
  auto* function = reinterpret_cast<std::function<void()>*>(data);
  (*function)();
}
} // namespace

void ConcreteAPI::streamHostFunction(void* streamPtr, const std::function<void()>& function) {
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);

  hipStreamCaptureStatus status{};
  APIWRAP(hipStreamIsCapturing(stream, &status));

  if (status != hipStreamCaptureStatusInvalidated) {
    auto* functionData = new std::function<void()>(function);
    if (status == hipStreamCaptureStatusActive) {
      APIWRAP(hipLaunchHostFunc(stream, &streamCallbackPermanent, functionData));
    } else {
      APIWRAP(hipLaunchHostFunc(stream, &streamCallbackEpheremal, functionData));
    }
  }
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
} // namespace

void ConcreteAPI::streamWaitMemory(void* streamPtr, uint32_t* location, uint32_t value) {
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
  uint32_t* deviceLocation = nullptr;
  APIWRAP(hipHostGetDevicePointer(reinterpret_cast<void**>(&deviceLocation), location, 0));
  const auto result = APIWRAPX(
      hipStreamWaitValue32(stream, deviceLocation, value, hipStreamWaitValueGte, 0xffffffff),
      {hipErrorNotSupported});
  if (result == hipErrorNotSupported) {
    spinloop<<<1, 1, 0, stream>>>(deviceLocation, value);
  }
}
