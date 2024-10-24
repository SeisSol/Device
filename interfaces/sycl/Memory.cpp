#include "SyclWrappedAPI.h"
#include <iostream>

using namespace device;

void *ConcreteAPI::allocGlobMem(size_t size) {
  auto *ptr = malloc_device(size, *this->currentDefaultQueue);
  this->currentStatistics->allocatedMemBytes += size;
  this->currentMemoryToSizeMap->insert({ptr, size});
  return ptr;
}

void *ConcreteAPI::allocUnifiedMem(size_t size, Destination hint) {
  auto *ptr = malloc_shared(size, *this->currentDefaultQueue);
  this->currentStatistics->allocatedUnifiedMemBytes += size;
  this->currentStatistics->allocatedMemBytes += size;
  this->currentMemoryToSizeMap->insert({ptr, size});
  return ptr;
}

void *ConcreteAPI::allocPinnedMem(size_t size, Destination hint) {
  auto *ptr = malloc_host(size, *this->currentDefaultQueue);
  this->currentStatistics->allocatedMemBytes += size;
  this->currentMemoryToSizeMap->insert({ptr, size});
  return ptr;
}

void ConcreteAPI::freeMem(void *devPtr) {
  // NOTE: Freeing nullptr results in segfault in oneAPI. It is an opposite behaviour
  // contrast to C++/CUDA/HIP
  if(devPtr != nullptr) {
    if (this->currentMemoryToSizeMap->find(devPtr) == this->currentMemoryToSizeMap->end())
      throw std::invalid_argument(this->getDeviceInfoAsText(this->currentDeviceId)
                                      .append("an attempt to delete memory that has not been allocated. Is this "
                                              "a pointer to this device or was this a double free?"));

    this->currentStatistics->deallocatedMemBytes += this->currentMemoryToSizeMap->at(devPtr);
    this->currentMemoryToSizeMap->erase(devPtr);
    free(devPtr, this->currentDefaultQueue->get_context());
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

char *ConcreteAPI::getStackMemory(size_t requestedBytes) {
  size_t requestedAlignedBytes = align(requestedBytes, getGlobMemAlignment());
  return this->currentDeviceStack->getStackMemory(requestedAlignedBytes);
}

void ConcreteAPI::popStackMemory() { this->currentDeviceStack->popStackMemory(); }

std::string ConcreteAPI::getMemLeaksReport() {
  std::ostringstream report{};

  report << "----MEMORY REPORT----\n";
  report << "Memory Leaks, bytes: "
         << (this->currentStatistics->allocatedMemBytes - this->currentStatistics->deallocatedMemBytes) << '\n';
  report << "Stack Memory Leaks, bytes: " << this->currentDeviceStack->getStackMemByteCounter() << '\n';
  report << "---------------------\n";

  return report.str();
}

size_t ConcreteAPI::getMaxAvailableMem() {
  auto device = this->currentDefaultQueue->get_device();
  return device.get_info<cl::sycl::info::device::global_mem_size>();
}

size_t ConcreteAPI::getCurrentlyOccupiedMem() { return this->currentStatistics->allocatedMemBytes; }

size_t ConcreteAPI::getCurrentlyOccupiedUnifiedMem() { return this->currentStatistics->allocatedUnifiedMemBytes; }

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
