// SPDX-FileCopyrightText: 2021 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Internals.h"
#include "SyclWrappedAPI.h"

#include <iostream>

using namespace device;
using namespace device::internals;

void* ConcreteAPI::allocGlobMem(size_t size, bool compress) {
  auto* ptr = malloc_device(size, this->currentDefaultQueue());
  this->currentStatistics().allocatedMemBytes += size;
  this->currentMemoryToSizeMap().insert({ptr, size});
  waitCheck(this->currentDefaultQueue());
  return ptr;
}

void* ConcreteAPI::allocUnifiedMem(size_t size, bool compress, Destination hint) {
  auto* ptr = malloc_shared(size, this->currentDefaultQueue());
  this->currentStatistics().allocatedUnifiedMemBytes += size;
  this->currentStatistics().allocatedMemBytes += size;
  this->currentMemoryToSizeMap().insert({ptr, size});
  waitCheck(this->currentDefaultQueue());
  return ptr;
}

void* ConcreteAPI::allocPinnedMem(size_t size, bool compress, Destination hint) {
  auto* ptr = malloc_host(size, this->currentDefaultQueue());
  this->currentStatistics().allocatedMemBytes += size;
  this->currentMemoryToSizeMap().insert({ptr, size});
  waitCheck(this->currentDefaultQueue());
  return ptr;
}

void ConcreteAPI::freeMem(void* devPtr) {
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

  // Use the first device context to free memory
  DeviceContext* context = this->availableDevices[0];
  if (!context)
    return;

  auto& map = context->memoryToSizeMap;

  if (map.find(devPtr) == map.end()) {
    return; // the std::throw is throwing some errors during the program finalization
  }

  context->statistics.deallocatedMemBytes += map.at(devPtr);
  map.erase(devPtr);
  auto& queue = context->queueBuffer.getDefaultQueue();
  sycl::free(devPtr, queue.get_context());
  queue.wait();
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
    this->freeMem(devPtr);
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
  } else {
    return malloc_device(size, *static_cast<sycl::queue*>(streamPtr));
  }
}
void ConcreteAPI::freeMemAsync(void* devPtr, void* streamPtr) {
  if (devPtr != nullptr) {
    free(devPtr, *static_cast<sycl::queue*>(streamPtr));
  }
}

std::string ConcreteAPI::getMemLeaksReport() {
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
  return this->currentStatistics().allocatedMemBytes;
}

size_t ConcreteAPI::getCurrentlyOccupiedUnifiedMem() {
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
