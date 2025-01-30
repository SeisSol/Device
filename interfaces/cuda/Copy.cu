// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "CudaWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"
#include <algorithm>
#include <cassert>

using namespace device;

void ConcreteAPI::copyTo(void *dst, const void *src, size_t count) {
  isFlagSet<DeviceSelected>(status);
  cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
  CHECK_ERR;
  statistics.explicitlyTransferredDataToDeviceBytes += count;
}

void ConcreteAPI::copyFrom(void *dst, const void *src, size_t count) {
  isFlagSet<DeviceSelected>(status);
  cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
  CHECK_ERR;
  statistics.explicitlyTransferredDataToHostBytes += count;
}

void ConcreteAPI::copyBetween(void *dst, const void *src, size_t count) {
  isFlagSet<DeviceSelected>(status);
  cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
  CHECK_ERR;
}

void ConcreteAPI::copyToAsync(void *dst, const void *src, size_t count, void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
  CHECK_ERR;
}

void ConcreteAPI::copyFromAsync(void *dst, const void *src, size_t count, void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);
  CHECK_ERR;
}

void ConcreteAPI::copyBetweenAsync(void *dst, const void *src, size_t count, void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
  CHECK_ERR;
}

void ConcreteAPI::copy2dArrayTo(void *dst, size_t dpitch, const void *src, size_t spitch,
                                size_t width, size_t height) {
  isFlagSet<DeviceSelected>(status);
  cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice);
  CHECK_ERR;
  statistics.explicitlyTransferredDataToDeviceBytes += width * height;
}

void ConcreteAPI::copy2dArrayFrom(void *dst, size_t dpitch, const void *src, size_t spitch,
                                  size_t width, size_t height) {
  isFlagSet<DeviceSelected>(status);
  cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost);
  CHECK_ERR;
  statistics.explicitlyTransferredDataToHostBytes += width * height;
}

void ConcreteAPI::prefetchUnifiedMemTo(Destination type, const void *devPtr, size_t count,
                                       void *streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  cudaStream_t stream = (streamPtr == nullptr) ? 0 : (static_cast<cudaStream_t>(streamPtr));
  cudaMemPrefetchAsync(devPtr,
                       count,
                       type == Destination::CurrentDevice ? currentDeviceId : cudaCpuDeviceId,
                       stream);
  CHECK_ERR;
}

