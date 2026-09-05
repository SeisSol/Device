// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "CudaWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"

#include <algorithm>
#include <cassert>
#include <functional>
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

  APIWRAP(cudaStreamSynchronize(defaultStream));
}

void* ConcreteAPI::createStream(double priority) {
  isFlagSet<InterfaceInitialized>(status);
  const std::lock_guard<std::mutex> lock(apiMutex);
  cudaStream_t stream;
  const auto truePriority = mapStreamPriority(priorityLeast, priorityGreatest, priority);
  APIWRAP(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, truePriority));
  genericStreams.insert(stream);
  return reinterpret_cast<void*>(stream);
}

void ConcreteAPI::destroyGenericStream(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  const std::lock_guard<std::mutex> lock(apiMutex);
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

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
  APIWRAP(cudaStreamDestroy(stream));
}

void ConcreteAPI::syncStreamWithHost(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

  CHECK_ERR;

  APIWRAP(cudaStreamSynchronize(stream));
}

bool ConcreteAPI::isStreamWorkDone(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

  const auto streamStatus = APIWRAPX(cudaStreamQuery(stream), {cudaErrorNotReady});

  return streamStatus == cudaSuccess;
}

void ConcreteAPI::syncStreamWithEvent(void* streamPtr, void* eventPtr) {
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
  cudaEvent_t event = static_cast<cudaEvent_t>(eventPtr);
  APIWRAP(cudaStreamWaitEvent(stream, event));
}

namespace {
// Called once, so the copy goes away with the call.
void streamCallbackEpheremal(void* data) {
  auto* function = reinterpret_cast<std::function<void()>*>(data);
  (*function)();
  delete function;
}

// Called on every replay of the graph it was recorded into, which owns the copy.
void streamCallbackRecorded(void* data) {
  auto* function = reinterpret_cast<std::function<void()>*>(data);
  (*function)();
}
} // namespace

void ConcreteAPI::streamHostFunction(void* streamPtr, const std::function<void()>& function) {
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

  const auto capture = internals::captureState(stream);
  if (capture.status == cudaStreamCaptureStatusInvalidated) {
    return;
  }

  if (capture.status == cudaStreamCaptureStatusActive) {
    auto* recorded = internals::adoptHostFunction(capture.graph, function);
    if (recorded == nullptr) {
      logError() << "A host function was recorded into a graph this backend does not know.";
      return;
    }
    APIWRAP(cudaLaunchHostFunc(stream, &streamCallbackRecorded, recorded));
    return;
  }

  APIWRAP(
      cudaLaunchHostFunc(stream, &streamCallbackEpheremal, new std::function<void()>(function)));
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
  // TODO: check for graph capture here?
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
  uint32_t* deviceLocation = nullptr;
  APIWRAP(cudaHostGetDevicePointer(&deviceLocation, location, 0));
  const auto result = DRVWRAPX(
      cuStreamWaitValue32(
          stream, reinterpret_cast<uintptr_t>(deviceLocation), value, CU_STREAM_WAIT_VALUE_GEQ),
      {CUDA_ERROR_NOT_SUPPORTED});
  if (result == CUDA_ERROR_NOT_SUPPORTED) {
    spinloop<<<1, 1, 0, stream>>>(deviceLocation, value);
  }
}
