// SPDX-FileCopyrightText: 2020Sol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "HipWrappedAPI.h"
#include "Internals.h"
#include "hip/hip_runtime.h"
#include "utils/logger.h"

#include <assert.h>
#include <iostream>
#include <sstream>

using namespace device;

void* ConcreteAPI::allocGlobMem(size_t size, bool compress) {
  isFlagSet<DeviceSelected>(status);
  void* devPtr;
  APIWRAP(hipMalloc(&devPtr, size));
  CHECK_ERR;
  statistics.allocatedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}

void* ConcreteAPI::allocUnifiedMem(size_t size, bool compress, Destination hint) {
  isFlagSet<DeviceSelected>(status);
  void* devPtr;
  APIWRAP(hipMallocManaged(&devPtr, size, hipMemAttachGlobal));
  if (hint == Destination::Host) {
    APIWRAP(hipMemAdvise(devPtr, size, hipMemAdviseSetPreferredLocation, hipCpuDeviceId));
  } else {
    APIWRAP(hipMemAdvise(devPtr, size, hipMemAdviseSetPreferredLocation, getDeviceId()));
  }
  CHECK_ERR;
  statistics.allocatedMemBytes += size;
  statistics.allocatedUnifiedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}

void* ConcreteAPI::allocPinnedMem(size_t size, bool compress, Destination hint) {
  isFlagSet<DeviceSelected>(status);
  void* devPtr;
  const auto flag = hint == Destination::Host ? hipHostMallocDefault : hipHostMallocMapped;
  APIWRAP(hipHostMalloc(&devPtr, size, flag));
  CHECK_ERR;
  statistics.allocatedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}

void ConcreteAPI::freeGlobMem(void* devPtr) {
  isFlagSet<DeviceSelected>(status);
  assert((memToSizeMap.find(devPtr) != memToSizeMap.end()) &&
         "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  statistics.deallocatedMemBytes += memToSizeMap[devPtr];
  APIWRAP(hipFree(devPtr));
  CHECK_ERR;
}

void ConcreteAPI::freeUnifiedMem(void* devPtr) {
  isFlagSet<DeviceSelected>(status);
  assert((memToSizeMap.find(devPtr) != memToSizeMap.end()) &&
         "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  statistics.deallocatedMemBytes += memToSizeMap[devPtr];
  APIWRAP(hipFree(devPtr));
  CHECK_ERR;
}

void ConcreteAPI::freePinnedMem(void* devPtr) {
  isFlagSet<DeviceSelected>(status);
  assert((memToSizeMap.find(devPtr) != memToSizeMap.end()) &&
         "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  statistics.deallocatedMemBytes += memToSizeMap[devPtr];
  APIWRAP(hipHostFree(devPtr));
  CHECK_ERR;
}

void* ConcreteAPI::allocMemAsync(size_t size, void* streamPtr) {
  if (size == 0) {
    return nullptr;
  } else {
    void* ptr;
    APIWRAP(hipMallocAsync(&ptr, size, static_cast<hipStream_t>(streamPtr)));
    CHECK_ERR;
    return ptr;
  }
}
void ConcreteAPI::freeMemAsync(void* devPtr, void* streamPtr) {
  if (devPtr != nullptr) {
    APIWRAP(hipFreeAsync(devPtr, static_cast<hipStream_t>(streamPtr)));
    CHECK_ERR;
  }
}

std::string ConcreteAPI::getMemLeaksReport() {
  isFlagSet<DeviceSelected>(status);
  std::ostringstream report{};
  report << "Memory Leaks, bytes: "
         << (statistics.allocatedMemBytes - statistics.deallocatedMemBytes) << '\n';
  return report.str();
}

size_t ConcreteAPI::getMaxAvailableMem() {
  isFlagSet<DeviceSelected>(status);
  hipDeviceProp_t property;
  APIWRAP(hipGetDeviceProperties(&property, getDeviceId()));
  CHECK_ERR;
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
  APIWRAP(hipHostRegister(ptr, size, hipHostRegisterDefault));
  CHECK_ERR;
}

void ConcreteAPI::unpinMemory(void* ptr) {
  isFlagSet<DeviceSelected>(status);
  APIWRAP(hipHostUnregister(ptr));
  CHECK_ERR;
}

void* ConcreteAPI::devicePointer(void* ptr) {
  isFlagSet<DeviceSelected>(status);
  void* result;
  APIWRAP(hipHostGetDevicePointer(&result, ptr, 0));
  CHECK_ERR;
  return result;
}
