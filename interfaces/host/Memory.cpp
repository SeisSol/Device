// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "HostWrappedAPI.h"
#include "utils/logger.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>

using namespace device;

void* ConcreteAPI::allocate(size_t size, bool unified) {
  isFlagSet<DeviceSelected>(status);
  if (size == 0) {
    return nullptr;
  }
  const size_t alignment = getGlobMemAlignment();
  // std::aligned_alloc requires the size to be a multiple of the alignment
  const size_t paddedSize = ((size + alignment - 1) / alignment) * alignment;
  void* ptr = std::aligned_alloc(alignment, paddedSize);
  if (ptr == nullptr) {
    logError() << "The host backend failed to allocate" << size << "bytes";
    return nullptr;
  }

  std::lock_guard guard(apiMutex);
  allocatedMemBytes += size;
  if (unified) {
    allocatedUnifiedMemBytes += size;
  }
  memToSizeMap[ptr] = size;
  return ptr;
}

void ConcreteAPI::deallocate(void* ptr) {
  isFlagSet<DeviceSelected>(status);
  if (ptr == nullptr) {
    return;
  }
  std::lock_guard guard(apiMutex);
  const auto it = memToSizeMap.find(ptr);
  if (it == memToSizeMap.end()) {
    logError() << "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer";
    return;
  }
  deallocatedMemBytes += it->second;
  memToSizeMap.erase(it);
  std::free(ptr);
}

void* ConcreteAPI::allocGlobMem(size_t size, bool /*compress*/) { return allocate(size, false); }

void* ConcreteAPI::allocUnifiedMem(size_t size, bool /*compress*/, Destination /*hint*/) {
  return allocate(size, true);
}

void* ConcreteAPI::allocPinnedMem(size_t size, bool /*compress*/, Destination /*hint*/) {
  return allocate(size, false);
}

void ConcreteAPI::freeGlobMem(void* devPtr) { deallocate(devPtr); }

void ConcreteAPI::freeUnifiedMem(void* devPtr) { deallocate(devPtr); }

void ConcreteAPI::freePinnedMem(void* devPtr) { deallocate(devPtr); }

std::string ConcreteAPI::getMemLeaksReport() {
  isFlagSet<DeviceSelected>(status);
  std::lock_guard guard(apiMutex);
  std::ostringstream report{};
  report << "Memory Leaks, bytes: " << (allocatedMemBytes - deallocatedMemBytes) << '\n';
  return report.str();
}

size_t ConcreteAPI::getCurrentlyOccupiedMem() {
  isFlagSet<DeviceSelected>(status);
  std::lock_guard guard(apiMutex);
  return allocatedMemBytes;
}

size_t ConcreteAPI::getCurrentlyOccupiedUnifiedMem() {
  isFlagSet<DeviceSelected>(status);
  std::lock_guard guard(apiMutex);
  return allocatedUnifiedMemBytes;
}

void* ConcreteAPI::allocMemAsync(size_t size, void* /*streamPtr*/) { return allocate(size, false); }

void ConcreteAPI::freeMemAsync(void* devPtr, void* /*streamPtr*/) { deallocate(devPtr); }

void ConcreteAPI::pinMemory(void* /*ptr*/, size_t /*size*/) {}

void ConcreteAPI::unpinMemory(void* /*ptr*/) {}

void* ConcreteAPI::devicePointer(void* ptr) { return ptr; }

void ConcreteAPI::copyTo(void* dst, const void* src, size_t count) { std::memcpy(dst, src, count); }

void ConcreteAPI::copyFrom(void* dst, const void* src, size_t count) {
  std::memcpy(dst, src, count);
}

void ConcreteAPI::copyBetween(void* dst, const void* src, size_t count) {
  std::memcpy(dst, src, count);
}

void ConcreteAPI::copyToAsync(void* dst, const void* src, size_t count, void* /*streamPtr*/) {
  std::memcpy(dst, src, count);
}

void ConcreteAPI::copyFromAsync(void* dst, const void* src, size_t count, void* /*streamPtr*/) {
  std::memcpy(dst, src, count);
}

void ConcreteAPI::copyBetweenAsync(void* dst, const void* src, size_t count, void* /*streamPtr*/) {
  std::memcpy(dst, src, count);
}

namespace {
void copy2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
  auto* dstBytes = static_cast<char*>(dst);
  const auto* srcBytes = static_cast<const char*>(src);
  for (size_t row = 0; row < height; ++row) {
    std::memcpy(dstBytes + row * dpitch, srcBytes + row * spitch, width);
  }
}
} // namespace

void ConcreteAPI::copy2dArrayTo(
    void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
  copy2d(dst, dpitch, src, spitch, width, height);
}

void ConcreteAPI::copy2dArrayFrom(
    void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
  copy2d(dst, dpitch, src, spitch, width, height);
}

void ConcreteAPI::prefetchUnifiedMemTo(Destination /*type*/,
                                       const void* /*devPtr*/,
                                       size_t /*count*/,
                                       void* /*streamPtr*/) {}
