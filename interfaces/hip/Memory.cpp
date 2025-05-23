// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "hip/hip_runtime.h"
#include <assert.h>
#include <sstream>
#include <iostream>

#include "HipWrappedAPI.h"
#include "Internals.h"

#include "utils/logger.h"

using namespace device;

void* ConcreteAPI::allocGlobMem(size_t size, bool compress) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  hipMalloc(&devPtr, size); CHECK_ERR;
  statistics.allocatedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}


void* ConcreteAPI::allocUnifiedMem(size_t size, bool compress, Destination hint) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  hipMallocManaged(&devPtr, size, hipMemAttachGlobal); CHECK_ERR;
  if (hint == Destination::Host) {
    hipMemAdvise(devPtr, size, hipMemAdviseSetPreferredLocation, hipCpuDeviceId);
    CHECK_ERR;
  }
  else {
    hipMemAdvise(devPtr, size, hipMemAdviseSetPreferredLocation, currentDeviceId);
    CHECK_ERR;
  }
  statistics.allocatedMemBytes += size;
  statistics.allocatedUnifiedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}


void* ConcreteAPI::allocPinnedMem(size_t size, bool compress, Destination hint) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  const auto flag = hint == Destination::Host ? hipHostMallocDefault : hipHostMallocMapped;
  hipHostMalloc(&devPtr, size, flag); CHECK_ERR;
  statistics.allocatedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}

void ConcreteAPI::freeGlobMem(void *devPtr) {
  isFlagSet<DeviceSelected>(status);
  assert((memToSizeMap.find(devPtr) != memToSizeMap.end()) 
         && "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  statistics.deallocatedMemBytes += memToSizeMap[devPtr];
  hipFree(devPtr); CHECK_ERR;
}

void ConcreteAPI::freeUnifiedMem(void *devPtr) {
  isFlagSet<DeviceSelected>(status);
  assert((memToSizeMap.find(devPtr) != memToSizeMap.end()) 
         && "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  statistics.deallocatedMemBytes += memToSizeMap[devPtr];
  hipFree(devPtr); CHECK_ERR;
}

void ConcreteAPI::freePinnedMem(void *devPtr) {
  isFlagSet<DeviceSelected>(status);
  assert((memToSizeMap.find(devPtr) != memToSizeMap.end())
         && "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  statistics.deallocatedMemBytes += memToSizeMap[devPtr];
  hipHostFree(devPtr); CHECK_ERR;
}

void *ConcreteAPI::allocMemAsync(size_t size, void* streamPtr) {
  if (size == 0) {
    return nullptr;
  }
  else {
    void* ptr;
    hipMallocAsync(&ptr, size, static_cast<hipStream_t>(streamPtr));
    CHECK_ERR;
    return ptr;
  }
}
void ConcreteAPI::freeMemAsync(void *devPtr, void* streamPtr) {
  if (devPtr != nullptr) {
    hipFreeAsync(devPtr, static_cast<hipStream_t>(streamPtr));
    CHECK_ERR;
  }
}

std::string ConcreteAPI::getMemLeaksReport() {
  isFlagSet<DeviceSelected>(status);
  std::ostringstream report{};
  report << "Memory Leaks, bytes: " << (statistics.allocatedMemBytes - statistics.deallocatedMemBytes) << '\n';
  return report.str();
}

size_t ConcreteAPI::getMaxAvailableMem() {
  isFlagSet<DeviceSelected>(status);
  hipDeviceProp_t property;
  hipGetDeviceProperties(&property, currentDeviceId); CHECK_ERR;
  return property.totalGlobalMem;
}

size_t ConcreteAPI::getCurrentlyOccupiedMem() {
  isFlagSet<DeviceSelected>(status);
  return statistics.allocatedMemBytes;
}

size_t ConcreteAPI::getCurrentlyOccupiedUnifiedMem() {
  isFlagSet<DeviceSelected>(status);
  return statistics.allocatedUnifiedMemBytes;
}

void ConcreteAPI::pinMemory(void* ptr, size_t size) {
  isFlagSet<DeviceSelected>(status);
  hipHostRegister(ptr, size, hipHostRegisterDefault);
  CHECK_ERR;
}

void ConcreteAPI::unpinMemory(void* ptr) {
  isFlagSet<DeviceSelected>(status);
  hipHostUnregister(ptr);
  CHECK_ERR;
}

void* ConcreteAPI::devicePointer(void* ptr) {
  isFlagSet<DeviceSelected>(status);
  void* result;
  hipHostGetDevicePointer(&result, ptr, 0);
  CHECK_ERR;
  return result;
}

