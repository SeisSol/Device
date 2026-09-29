// SPDX-FileCopyrightText: 2021 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Internals.h"
#include "SyclWrappedAPI.h"

#include <iostream>
#include <mutex>
#include <sycl/sycl.hpp>

#ifdef SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif

using namespace device;
using namespace device::internals;

void* ConcreteAPI::allocGlobMem(size_t size, bool compress) {
  const std::lock_guard<std::mutex> lock(apiMutex);

  auto* ptr = malloc_device(size, this->currentDefaultQueue());
  this->currentStatistics().allocatedMemBytes += size;
  this->currentMemoryToSizeMap().insert({ptr, size});
  return ptr;
}

void* ConcreteAPI::allocUnifiedMem(size_t size, bool compress, Destination hint) {
  const std::lock_guard<std::mutex> lock(apiMutex);

  auto* ptr = malloc_shared(size, this->currentDefaultQueue());
  this->currentStatistics().allocatedUnifiedMemBytes += size;
  this->currentStatistics().allocatedMemBytes += size;
  this->currentMemoryToSizeMap().insert({ptr, size});
  return ptr;
}

void* ConcreteAPI::allocPinnedMem(size_t size, bool compress, Destination hint) {
  const std::lock_guard<std::mutex> lock(apiMutex);

  auto* ptr = malloc_host(size, this->currentDefaultQueue());
  this->currentStatistics().allocatedMemBytes += size;
  this->currentMemoryToSizeMap().insert({ptr, size});
  return ptr;
}

void ConcreteAPI::freeMem(void* devPtr, bool unified) {
  // NOTE: Freeing nullptr results in segfault in oneAPI. It is an opposite behaviour
  // contrast to C++/CUDA/HIP
  if (devPtr == nullptr) {
    return;
  }

  if (!this->deviceInitialized) {
    return;
  }

  if (this->availableDevices.empty()) {
    return;
  }

  const std::lock_guard<std::mutex> lock(apiMutex);

  // Use the first device context to free memory
  DeviceContext* context = this->availableDevices[getDeviceId()];
  if (!context) {
    return;
  }
  auto& map = context->memoryToSizeMap;

  if (map.find(devPtr) == map.end()) {
    return; // the std::throw is throwing some errors during the program finalization
  }

  const auto size = map.at(devPtr);
  context->statistics.deallocatedMemBytes += size;
  if (unified) {
    context->statistics.allocatedUnifiedMemBytes -= size;
  }
  map.erase(devPtr);
  // freeing memory that a queue may still be reading from is undefined, and the caller has no
  // way to state that it is done, so the wait stays
  auto& queue = context->queueBuffer.getDefaultQueue();
  queue.wait();
  sycl::free(devPtr, queue.get_context());
}

void ConcreteAPI::freeGlobMem(void* devPtr) {
  // NOTE: Freeing nullptr results in segfault in oneAPI. It is an opposite behavior
  // contrast to C++/CUDA/HIP
  if (devPtr != nullptr) {
    this->freeMem(devPtr);
  }
}

void ConcreteAPI::freeUnifiedMem(void* devPtr) {
  // NOTE: Freeing nullptr results in segfault in oneAPI. It is an opposite behavior
  // contrast to C++/CUDA/HIP
  if (devPtr != nullptr) {
    this->freeMem(devPtr, true);
  }
}

void ConcreteAPI::freePinnedMem(void* devPtr) {
  // NOTE: Freeing nullptr results in segfault in oneAPI. It is an opposite behavior
  // contrast to C++/CUDA/HIP
  if (devPtr != nullptr) {
    this->freeMem(devPtr);
  }
}

void* ConcreteAPI::allocMemAsync(size_t size, void* streamPtr) {
  if (size == 0) {
    return nullptr;
  }

  auto& queue = *static_cast<sycl::queue*>(streamPtr);
#ifdef SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
  if (this->currentContext()->asyncMemoryAlloc) {
    return sycl::ext::oneapi::experimental::async_malloc(queue, sycl::usm::alloc::device, size);
  }
#endif
  return malloc_device(size, queue);
}

void ConcreteAPI::freeMemAsync(void* devPtr, void* streamPtr) {
  if (devPtr == nullptr) {
    return;
  }

  auto& queue = *static_cast<sycl::queue*>(streamPtr);
#ifdef SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
  if (this->currentContext()->asyncMemoryAlloc) {
    sycl::ext::oneapi::experimental::async_free(queue, devPtr);
    return;
  }
#endif
  // Without a stream-ordered free, sycl::free releases the memory right away, while the work
  // enqueued before may still use it - with AdaptiveCpp, that work may not even have been
  // submitted yet. Hence, wait for it first.
  this->currentQueueBuffer().syncQueueWithHost(&queue);
  free(devPtr, queue);
}

std::string ConcreteAPI::getMemLeaksReport() {
  const std::lock_guard<std::mutex> lock(apiMutex);

  std::ostringstream report{};

  report << "----MEMORY REPORT----\n";
  report << "Memory Leaks, bytes: "
         << (this->currentStatistics().allocatedMemBytes -
             this->currentStatistics().deallocatedMemBytes)
         << '\n';
  report << "---------------------\n";

  return report.str();
}

size_t ConcreteAPI::getMaxAvailableMem() {
  auto device = this->currentDefaultQueue().get_device();
  return device.get_info<sycl::info::device::global_mem_size>();
}

size_t ConcreteAPI::getCurrentlyOccupiedMem() {
  const std::lock_guard<std::mutex> lock(apiMutex);

  return this->currentStatistics().allocatedMemBytes;
}

size_t ConcreteAPI::getCurrentlyOccupiedUnifiedMem() {
  const std::lock_guard<std::mutex> lock(apiMutex);

  return this->currentStatistics().allocatedUnifiedMemBytes;
}

void ConcreteAPI::pinMemory(void* ptr, size_t size) {
  // not supported
  throw std::exception();
}

void ConcreteAPI::unpinMemory(void* ptr) {
  // not supported
  throw std::exception();
}

void* ConcreteAPI::devicePointer(void* ptr) { return ptr; }
