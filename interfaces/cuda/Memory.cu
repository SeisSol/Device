#include <assert.h>
#include <iostream>
#include <sstream>

#include "CudaWrappedAPI.h"
#include "Internals.h"

using namespace device;

void *ConcreteAPI::allocGlobMem(size_t size) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  cudaMalloc(&devPtr, size);
  CHECK_ERR;
  statistics.allocatedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}

void *ConcreteAPI::allocUnifiedMem(size_t size) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  cudaMallocManaged(&devPtr, size, cudaMemAttachGlobal);
  CHECK_ERR;
  statistics.allocatedMemBytes += size;
  statistics.allocatedUnifiedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}

void *ConcreteAPI::allocPinnedMem(size_t size) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  cudaMallocHost(&devPtr, size);
  CHECK_ERR;
  statistics.allocatedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}

void ConcreteAPI::freeMem(void *devPtr) {
  isFlagSet<DeviceSelected>(status);
  assert((memToSizeMap.find(devPtr) != memToSizeMap.end()) &&
         "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  statistics.deallocatedMemBytes += memToSizeMap[devPtr];
  cudaFree(devPtr);
  CHECK_ERR;
}

void ConcreteAPI::freePinnedMem(void *devPtr) {
  isFlagSet<DeviceSelected>(status);
  assert((memToSizeMap.find(devPtr) != memToSizeMap.end()) &&
         "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  statistics.deallocatedMemBytes += memToSizeMap[devPtr];
  cudaFreeHost(devPtr);
  CHECK_ERR;
}

char *ConcreteAPI::getStackMemory(size_t requestedBytes) {
  isFlagSet<StackMemAllocated>(status);
  assert(((stackMemByteCounter + requestedBytes) < maxStackMem) &&
         "DEVICE:: run out of a device stack memory");
  char *mem = &stackMemory[stackMemByteCounter];

  size_t requestedAlignedBytes = align(requestedBytes, getGlobMemAlignment());
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
  report << "Memory Leaks, bytes: "
         << (statistics.allocatedMemBytes - statistics.deallocatedMemBytes) << '\n';
  report << "Stack Memory Leaks, bytes: " << stackMemByteCounter << '\n';
  return report.str();
}

size_t ConcreteAPI::getMaxAvailableMem() {
  isFlagSet<DeviceSelected>(status);
  cudaDeviceProp property;
  cudaGetDeviceProperties(&property, currentDeviceId);
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