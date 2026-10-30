// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "HipWrappedAPI.h"
#include "Internals.h"
#include "OwnQueues.h"
#include "hip/hip_runtime.h"
#include "utils/logger.h"

#include <algorithm>

using namespace device;

void ConcreteAPI::copyTo(void* dst, const void* src, size_t count) {
  isFlagSet<DeviceSelected>(status);
  APIWRAP(hipMemcpy(dst, src, count, hipMemcpyHostToDevice));
  statistics.explicitlyTransferredDataToDeviceBytes += count;
}

void ConcreteAPI::copyFrom(void* dst, const void* src, size_t count) {
  isFlagSet<DeviceSelected>(status);
  APIWRAP(hipMemcpy(dst, src, count, hipMemcpyDeviceToHost));
  statistics.explicitlyTransferredDataToHostBytes += count;
}

void ConcreteAPI::copyBetween(void* dst, const void* src, size_t count) {
  isFlagSet<DeviceSelected>(status);
  APIWRAP(hipMemcpy(dst, src, count, hipMemcpyDeviceToDevice));
}

void ConcreteAPI::copyToAsync(void* dst, const void* src, size_t count, void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  if (ownqueues::isTagged(streamPtr) && !ownqueues::capturing(streamPtr)) {
    ownqueues::copy(dst, src, count, streamPtr);
    return;
  }
  auto stream = (streamPtr != nullptr) ? ownqueues::vehicle(streamPtr) : 0;
  APIWRAP(hipMemcpyAsync(dst, src, count, hipMemcpyHostToDevice, stream));
}

void ConcreteAPI::copyFromAsync(void* dst, const void* src, size_t count, void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  if (ownqueues::isTagged(streamPtr) && !ownqueues::capturing(streamPtr)) {
    ownqueues::copy(dst, src, count, streamPtr);
    return;
  }
  auto stream = (streamPtr != nullptr) ? ownqueues::vehicle(streamPtr) : 0;
  APIWRAP(hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToHost, stream));
}

void ConcreteAPI::copyBetweenAsync(void* dst, const void* src, size_t count, void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  if (ownqueues::isTagged(streamPtr) && !ownqueues::capturing(streamPtr)) {
    ownqueues::copy(dst, src, count, streamPtr);
    return;
  }
  auto stream = (streamPtr != nullptr) ? ownqueues::vehicle(streamPtr) : 0;
  APIWRAP(hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToDevice, stream));
}

void ConcreteAPI::copy2dArrayTo(
    void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
  isFlagSet<DeviceSelected>(status);
  APIWRAP(hipMemcpy2D(dst, dpitch, src, spitch, width, height, hipMemcpyHostToDevice));
  statistics.explicitlyTransferredDataToDeviceBytes += width * height;
}

void ConcreteAPI::copy2dArrayFrom(
    void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
  isFlagSet<DeviceSelected>(status);
  APIWRAP(hipMemcpy2D(dst, dpitch, src, spitch, width, height, hipMemcpyDeviceToHost));
  statistics.explicitlyTransferredDataToHostBytes += width * height;
}

void ConcreteAPI::prefetchUnifiedMemTo(Destination type,
                                       const void* devPtr,
                                       size_t count,
                                       void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  // (a hint, which the stream standing in for an own one may take in any order)
  hipStream_t stream = (streamPtr == nullptr) ? 0 : ownqueues::vehicle(streamPtr);
  APIWRAP(hipMemPrefetchAsync(
      devPtr, count, type == Destination::CurrentDevice ? getDeviceId() : hipCpuDeviceId, stream));
}
