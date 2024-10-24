#include "hip/hip_runtime.h"
#include <assert.h>
#include <sstream>
#include <iostream>

#include "HipWrappedAPI.h"
#include "Internals.h"

using namespace device;

void* ConcreteAPI::allocGlobMem(size_t size) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  hipMalloc(&devPtr, size); CHECK_ERR;
  statistics.allocatedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}


void* ConcreteAPI::allocUnifiedMem(size_t size) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  hipMallocManaged(&devPtr, size, hipMemAttachGlobal); CHECK_ERR;
  statistics.allocatedMemBytes += size;
  statistics.allocatedUnifiedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}


void* ConcreteAPI::allocPinnedMem(size_t size) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  hipHostMalloc(&devPtr, size, hipHostMallocDefault); CHECK_ERR;
  statistics.allocatedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}


void ConcreteAPI::freeMem(void *devPtr) {
  isFlagSet<DeviceSelected>(status);
  assert((memToSizeMap.find(devPtr) != memToSizeMap.end()) 
         && "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  statistics.deallocatedMemBytes += memToSizeMap[devPtr];
  hipFree(devPtr); CHECK_ERR;
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


char* ConcreteAPI::getStackMemory(size_t requestedBytes) {
  isFlagSet<StackMemAllocated>(status);
  char *mem = &stackMemory[stackMemByteCounter];

  size_t requestedAlignedBytes = align(requestedBytes, getGlobMemAlignment());

  if ((stackMemByteCounter + requestedAlignedBytes) < maxStackMem) {
    logError() << "DEVICE:: run out of a device stack memory";
  }

  stackMemByteCounter += requestedAlignedBytes;
  stackMemMeter.push(requestedAlignedBytes);
  return mem;
}


void ConcreteAPI::popStackMemory() {
  isFlagSet<StackMemAllocated>(status);
  stackMemByteCounter -= stackMemMeter.top();
  stackMemMeter.pop();
}

std::string ConcreteAPI::getMemLeaksReport() {
  isFlagSet<DeviceSelected>(status);
  std::ostringstream report{};
  report << "Memory Leaks, bytes: " << (statistics.allocatedMemBytes - statistics.deallocatedMemBytes) << '\n';
  report << "Stack Memory Leaks, bytes: " << stackMemByteCounter << '\n';
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
