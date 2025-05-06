// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "SyclWrappedAPI.h"

#include <algorithm>
#include <math.h>

using namespace device;

void ConcreteAPI::copyTo(void *dst, const void *src, size_t size) {
  this->currentStatistics->explicitlyTransferredDataToDeviceBytes += size;
  this->currentDefaultQueue->submit([&](sycl::handler &cgh) { cgh.memcpy(dst, src, size); }).wait_and_throw();
}

void ConcreteAPI::copyFrom(void *dst, const void *src, size_t size) {
  this->currentStatistics->explicitlyTransferredDataToHostBytes += size;
  this->currentDefaultQueue->submit([&](sycl::handler &cgh) { cgh.memcpy(dst, src, size); }).wait_and_throw();
}

void ConcreteAPI::copyBetween(void *dst, const void *src, size_t size) {
  this->currentDefaultQueue->submit([&](sycl::handler &cgh) { cgh.memcpy(dst, src, size); }).wait_and_throw();
}

void ConcreteAPI::copy2dArrayTo(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) {
  if (width > dpitch || width > spitch)
    throw std::invalid_argument("width exceeds row pitch of matrix");

  auto diffPitch = std::max(spitch, dpitch) - std::min(spitch, dpitch);
  for (size_t i = 0; i < height; i++) {
    this->copyTo((std::byte *)dst + i * dpitch, (std::byte *)src + i * spitch, width + diffPitch);
  }
}

void ConcreteAPI::copy2dArrayFrom(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                                  size_t height) {
  if (width > dpitch || width > spitch)
    throw std::invalid_argument("width exceeds row pitch of matrix");

  auto diffPitch = std::max(spitch, dpitch) - std::min(spitch, dpitch);
  for (size_t i = 0; i < height; i++) {
    this->copyFrom((std::byte *)dst + i * dpitch, (std::byte *)src + i * spitch, width + diffPitch);
  }
}

void ConcreteAPI::copyToAsync(void *dst, const void *src, size_t count, void *streamPtr) {
  auto *targetQueue = (streamPtr != nullptr) ? static_cast<sycl::queue *>(streamPtr) : this->currentDefaultQueue;
  if (!this->currentQueueBuffer->exists(targetQueue))
    throw std::invalid_argument(getDeviceInfoAsText(currentDeviceId)
                                    .append("tried to prefetch usm on a queue that is not known to this device"));

  targetQueue->submit([&](sycl::handler &cgh) { cgh.memcpy(dst, src, count); });
}

void ConcreteAPI::copyFromAsync(void *dst, const void *src, size_t count, void *streamPtr) {
  this->copyToAsync(dst, src, count, streamPtr);
}

void ConcreteAPI::copyBetweenAsync(void *dst, const void *src, size_t count, void *streamPtr) {
  this->copyToAsync(dst, src, count, streamPtr);
}

void ConcreteAPI::prefetchUnifiedMemTo(Destination type, const void *devPtr, size_t count, void *streamPtr) {
  auto *asQueue = (streamPtr != nullptr) ? static_cast<sycl::queue *>(streamPtr) : this->currentDefaultQueue;
  if (!this->currentQueueBuffer->exists(asQueue))
    throw std::invalid_argument(getDeviceInfoAsText(currentDeviceId)
                                    .append("tried to prefetch usm on a queue that is not known to this device"));

  asQueue->submit([&](sycl::handler &cgh) { cgh.prefetch(devPtr, count); });
}

