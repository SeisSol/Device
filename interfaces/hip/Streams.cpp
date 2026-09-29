// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "HipWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"

#include <algorithm>
#include <cassert>
#include <mutex>
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
  const std::lock_guard<std::mutex> lock(apiMutex);
  hipStream_t stream;
  const auto truePriority = mapStreamPriority(priorityLeast, priorityGreatest, priority);
  APIWRAP(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, truePriority));
  genericStreams.insert(stream);
  return reinterpret_cast<void*>(stream);
}

void ConcreteAPI::destroyGenericStream(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  const std::lock_guard<std::mutex> lock(apiMutex);
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);

  // The stream has to leave the set before it is destroyed, and a stream that is not in it is not
  // this backend's to destroy - the default stream, for one, would take the whole interface with
  // it.
  auto it = genericStreams.find(stream);
  if (it == genericStreams.end()) {
    logWarning() << "Tried to destroy a stream that this device does not know about. It has "
                    "either been destroyed already or was not created here; not destroying it.";
    return;
  }

  genericStreams.erase(it);
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
// Called once, so the copy goes away with the call.
void streamCallbackEpheremal(void* data) {
  auto* function = reinterpret_cast<std::function<void()>*>(data);
  (*function)();
  delete function;
}

void streamCallbackRecorded(void* data) {
  auto* function = reinterpret_cast<std::function<void()>*>(data);
  (*function)();
}
} // namespace

void ConcreteAPI::streamHostFunction(void* streamPtr, const std::function<void()>& function) {
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);

  const auto capture = internals::captureState(stream);
  if (capture.status == hipStreamCaptureStatusInvalidated) {
    return;
  }

  if (capture.status == hipStreamCaptureStatusActive) {
    auto* recorded = internals::adoptHostFunction(capture.graph, function);
    if (recorded == nullptr) {
      logError() << "A host function was recorded into a graph this backend does not know.";
      return;
    }
    APIWRAP(hipLaunchHostFunc(stream, &streamCallbackRecorded, recorded));
    return;
  }

  APIWRAP(hipLaunchHostFunc(stream, &streamCallbackEpheremal, new std::function<void()>(function)));
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
