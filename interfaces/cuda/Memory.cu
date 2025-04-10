// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include <assert.h>
#include <driver_types.h>
#include <iostream>
#include <sstream>

#include "AbstractAPI.h"
#include "CudaWrappedAPI.h"
#include "Internals.h"

#include "utils/logger.h"

using namespace device;

void *ConcreteAPI::allocGlobMem(size_t size, bool compress) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  cudaMalloc(&devPtr, size);
  CHECK_ERR;
  statistics.allocatedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}

void *ConcreteAPI::allocUnifiedMem(size_t size, bool compress, Destination hint) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  cudaMallocManaged(&devPtr, size, cudaMemAttachGlobal);
  CHECK_ERR;
  if (hint == Destination::Host) {
    cudaMemAdvise(devPtr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    CHECK_ERR;
  }
  else if (allowedConcurrentManagedAccess) {
    cudaMemAdvise(devPtr, size, cudaMemAdviseSetPreferredLocation, currentDeviceId);
    CHECK_ERR;
  }
  statistics.allocatedMemBytes += size;
  statistics.allocatedUnifiedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}

void *ConcreteAPI::allocPinnedMem(size_t size, bool compress, Destination hint) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  const auto flag = hint == Destination::Host ? cudaHostAllocDefault : cudaHostAllocMapped;
  cudaHostAlloc(&devPtr, size, flag);
  CHECK_ERR;
  statistics.allocatedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}

void ConcreteAPI::freeGlobMem(void *devPtr) {
  isFlagSet<DeviceSelected>(status);
  assert((memToSizeMap.find(devPtr) != memToSizeMap.end()) &&
         "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  statistics.deallocatedMemBytes += memToSizeMap[devPtr];
  cudaFree(devPtr);
  CHECK_ERR;
}

void ConcreteAPI::freeUnifiedMem(void *devPtr) {
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
  char *mem = &stackMemory[stackMemByteCounter];

  size_t requestedAlignedBytes = align(requestedBytes, getGlobMemAlignment());

  if ((stackMemByteCounter + requestedAlignedBytes) >= maxStackMem) {
    logError() << "DEVICE:: run out of device stack memory";
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

void ConcreteAPI::pinMemory(void* ptr, size_t size) {
  isFlagSet<DeviceSelected>(status);
  cudaHostRegister(ptr, size, 0);
  CHECK_ERR;
}

void ConcreteAPI::unpinMemory(void* ptr) {
  isFlagSet<DeviceSelected>(status);
  cudaHostUnregister(ptr);
  CHECK_ERR;
}

void* ConcreteAPI::devicePointer(void* ptr) {
  isFlagSet<DeviceSelected>(status);
  void* result;
  cudaHostGetDevicePointer(&result, ptr, 0);
  CHECK_ERR;
  return result;
}

