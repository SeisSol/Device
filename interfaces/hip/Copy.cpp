// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "HipWrappedAPI.h"
#include "Internals.h"
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
  auto stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  APIWRAP(hipMemcpyAsync(dst, src, count, hipMemcpyHostToDevice, stream));
}

void ConcreteAPI::copyFromAsync(void* dst, const void* src, size_t count, void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  auto stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  APIWRAP(hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToHost, stream));
}

void ConcreteAPI::copyBetweenAsync(void* dst, const void* src, size_t count, void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  auto stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
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
  hipStream_t stream = (streamPtr == nullptr) ? 0 : (static_cast<hipStream_t>(streamPtr));
  APIWRAP(hipMemPrefetchAsync(
      devPtr, count, type == Destination::CurrentDevice ? getDeviceId() : hipCpuDeviceId, stream));
}
