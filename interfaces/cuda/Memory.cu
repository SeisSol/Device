// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include <assert.h>
#include <cstring>
#include <driver_types.h>
#include <iostream>
#include <sstream>

#include "AbstractAPI.h"
#include "CudaWrappedAPI.h"
#include "Internals.h"

#include "utils/logger.h"

#include <cuda.h>

using namespace device;

namespace {

// adapted from https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp

void* driverAllocate(std::size_t size, const CUmemAllocationProp& prop) {
  std::size_t granularity = 1;
  cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  CHECK_ERR;

  size = ((size + granularity - 1) / granularity) * granularity;

  CUdeviceptr cptr;
  cuMemAddressReserve(&cptr, size, 0, 0, 0);
  CHECK_ERR;
  // cf. https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?highlight=inline%20compression#inline-compression
  CUmemGenericAllocationHandle allocationHandle;
  cuMemCreate(&allocationHandle, size, &prop, 0);
  CHECK_ERR;
  cuMemMap(cptr, size, 0, allocationHandle, 0);
  CHECK_ERR;
  cuMemRelease(allocationHandle);
  CHECK_ERR;

  CUmemAccessDesc accessDescriptor{};
  accessDescriptor.location.id = prop.location.id;
  accessDescriptor.location.type = prop.location.type;
  accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  cuMemSetAccess(cptr, size, &accessDescriptor, 1);

  return reinterpret_cast<void*>(cptr);
}

void driverFree(void* ptr, std::size_t size, const CUmemAllocationProp& prop) {
  std::size_t granularity = 1;
  cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  CHECK_ERR;

  size = ((size + granularity - 1) / granularity) * granularity;

  CUdeviceptr cptr = reinterpret_cast<CUdeviceptr>(ptr);

  cuMemUnmap(cptr, size);
  CHECK_ERR;
  cuMemAddressFree(cptr, size);
  CHECK_ERR;
}

} // namespace

void *ConcreteAPI::allocGlobMem(size_t size, bool compress) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  if (compress && canCompress) {
    CUmemAllocationProp prop = {};
    std::memset(&prop, 0, sizeof(CUmemAllocationProp));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = getDeviceId();
    prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    devPtr = driverAllocate(size, prop);
    allocationProperties[devPtr] = reinterpret_cast<void*>(new CUmemAllocationProp(prop));
  }
  else {
    cudaMalloc(&devPtr, size);
    CHECK_ERR;
  }
  statistics.allocatedMemBytes += size;
  memToSizeMap[devPtr] = size;
  return devPtr;
}

void *ConcreteAPI::allocUnifiedMem(size_t size, bool compress, Destination hint) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  cudaMallocManaged(&devPtr, size, cudaMemAttachGlobal);
  CHECK_ERR;

  cudaMemLocation location{};
  if (hint == Destination::Host) {
    location.id = cudaCpuDeviceId;
#if CUDART_VERSION >= 13000
    location.type = cudaMemLocationTypeHost;
#endif
  }
  else if (allowedConcurrentManagedAccess) {
    location.id = getDeviceId();
#if CUDART_VERSION >= 13000
    location.type = cudaMemLocationTypeDevice;
#endif
  }

  cudaMemAdvise(devPtr, size, cudaMemAdviseSetPreferredLocation,
#if CUDART_VERSION >= 13000
                       location
#else
                       location.id
#endif
  );
  CHECK_ERR;

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
  if (allocationProperties.find(devPtr) != allocationProperties.end()) {
    driverFree(devPtr, memToSizeMap.at(devPtr), *reinterpret_cast<CUmemAllocationProp*>(allocationProperties.at(devPtr)));
  }
  else {
    cudaFree(devPtr);
    CHECK_ERR;
  }
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

void *ConcreteAPI::allocMemAsync(size_t size, void* streamPtr) {
  if (size == 0) {
    return nullptr;
  }
  else {
    void* ptr;
    cudaMallocAsync(&ptr, size, static_cast<cudaStream_t>(streamPtr));
    CHECK_ERR;
    return ptr;
  }
}
void ConcreteAPI::freeMemAsync(void *devPtr, void* streamPtr) {
  if (devPtr != nullptr) {
    cudaFreeAsync(devPtr, static_cast<cudaStream_t>(streamPtr));
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
  cudaDeviceProp property;
  cudaGetDeviceProperties(&property, getDeviceId());
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

