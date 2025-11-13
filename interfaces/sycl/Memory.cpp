// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "SyclWrappedAPI.h"
#include <iostream>

using namespace device;

void *ConcreteAPI::allocGlobMem(size_t size, bool compress) {
  auto *ptr = malloc_device(size, this->currentDefaultQueue());
  this->currentStatistics().allocatedMemBytes += size;
  this->currentMemoryToSizeMap().insert({ptr, size});
  this->currentDefaultQueue().wait();
  return ptr;
}

void *ConcreteAPI::allocUnifiedMem(size_t size, bool compress, Destination hint) {
  auto *ptr = malloc_shared(size, this->currentDefaultQueue());
  this->currentStatistics().allocatedUnifiedMemBytes += size;
  this->currentStatistics().allocatedMemBytes += size;
  this->currentMemoryToSizeMap().insert({ptr, size});
  this->currentDefaultQueue().wait();
  return ptr;
}

void *ConcreteAPI::allocPinnedMem(size_t size, bool compress, Destination hint) {
  auto *ptr = malloc_host(size, this->currentDefaultQueue());
  this->currentStatistics().allocatedMemBytes += size;
  this->currentMemoryToSizeMap().insert({ptr, size});
  this->currentDefaultQueue().wait();
  return ptr;
}

void ConcreteAPI::freeMem(void *devPtr) {
  // NOTE: Freeing nullptr results in segfault in oneAPI. It is an opposite behaviour
  // contrast to C++/CUDA/HIP
  if(devPtr != nullptr) {
    if (this->currentMemoryToSizeMap().find(devPtr) == this->currentMemoryToSizeMap().end()) {
      throw std::invalid_argument(this->getDeviceInfoAsText(getDeviceId())
                                      .append("an attempt to delete memory that has not been allocated. Is this "
                                              "a pointer to this device or was this a double free?"));
    }

    this->currentStatistics().deallocatedMemBytes += this->currentMemoryToSizeMap().at(devPtr);
    this->currentMemoryToSizeMap().erase(devPtr);
    free(devPtr, this->currentDefaultQueue().get_context());
    this->currentDefaultQueue().wait();
  }
}

void ConcreteAPI::freeGlobMem(void *devPtr) {
  // NOTE: Freeing nullptr results in segfault in oneAPI. It is an opposite behavior
  // contrast to C++/CUDA/HIP
  if(devPtr != nullptr) {
    this->freeMem(devPtr);
  }
}

void ConcreteAPI::freeUnifiedMem(void *devPtr) {
  // NOTE: Freeing nullptr results in segfault in oneAPI. It is an opposite behavior
  // contrast to C++/CUDA/HIP
  if(devPtr != nullptr) {
    this->freeMem(devPtr);
  }
}

void ConcreteAPI::freePinnedMem(void *devPtr) {
  // NOTE: Freeing nullptr results in segfault in oneAPI. It is an opposite behavior
  // contrast to C++/CUDA/HIP
  if(devPtr != nullptr) {
    this->freeMem(devPtr);
  }
}

void *ConcreteAPI::allocMemAsync(size_t size, void* streamPtr) {
  if (size == 0) {
    return nullptr;
  }
  else {
    return malloc_device(size, *static_cast<sycl::queue*>(streamPtr));
  }
}
void ConcreteAPI::freeMemAsync(void *devPtr, void* streamPtr) {
  if (devPtr != nullptr) {
    free(devPtr, *static_cast<sycl::queue*>(streamPtr));
  }
}

std::string ConcreteAPI::getMemLeaksReport() {
  std::ostringstream report{};

  report << "----MEMORY REPORT----\n";
  report << "Memory Leaks, bytes: "
         << (this->currentStatistics().allocatedMemBytes - this->currentStatistics().deallocatedMemBytes) << '\n';
  report << "---------------------\n";

  return report.str();
}

size_t ConcreteAPI::getMaxAvailableMem() {
  auto device = this->currentDefaultQueue().get_device();
  return device.get_info<sycl::info::device::global_mem_size>();
}

size_t ConcreteAPI::getCurrentlyOccupiedMem() { return this->currentStatistics().allocatedMemBytes; }

size_t ConcreteAPI::getCurrentlyOccupiedUnifiedMem() { return this->currentStatistics().allocatedUnifiedMemBytes; }

void ConcreteAPI::pinMemory(void* ptr, size_t size) {
  // not supported
  throw std::exception();
}

void ConcreteAPI::unpinMemory(void* ptr) {
  // not supported
  throw std::exception();
}

void* ConcreteAPI::devicePointer(void* ptr) {
  return ptr;
}

