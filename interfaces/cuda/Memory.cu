#include <assert.h>
#include <iostream>
#include <sstream>
#include <cstring>
#include <stdexcept>

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

namespace device {
cudaError_t setProp(CUmemAllocationProp *prop) {
  CUdevice currentDevice;
  if (cuCtxGetDevice(&currentDevice) != CUDA_SUCCESS)
    return cudaErrorMemoryAllocation;

  std::memset(prop, 0, sizeof(CUmemAllocationProp));
  prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop->location.id = currentDevice;
  prop->allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

  return cudaSuccess;
}
}

void *ConcreteAPI::allocCompressibleGlobMem(size_t size) {
  void* addr{ nullptr};
  if (useCompressibleMemory) {
    isFlagSet<DeviceSelected>(status);
    try {
      std::stringstream errStream;

      CUmemAllocationProp prop = {};
      cudaError_t err = setProp(&prop);

      if (err != cudaSuccess) {
        errStream << "could not set device properties";
        throw std::runtime_error(errStream.str());
      }

      size_t granularity = 0;
      CUresult result = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
      if (result != CUDA_SUCCESS) {
        errStream << "could not calculates granularity. Error: " << result;
        throw std::runtime_error(std::to_string(result));
      }
      size = ((size - 1) / granularity + 1) * granularity;

      CUdeviceptr dptr;
      result = cuMemAddressReserve(&dptr, size, 0, 0, 0);
      if (result != CUDA_SUCCESS) {
        errStream << "could not reserve memory address. Error: " << result;
        throw std::runtime_error(errStream.str());
      }

      CUmemGenericAllocationHandle allocationHandle;
      result = cuMemCreate(&allocationHandle, size, &prop, 0);
      if (result != CUDA_SUCCESS) {
        errStream << "could not create compressible memory. Error: " << result;
        throw std::runtime_error(errStream.str());
      }

      CUmemAllocationProp allocationProp = {};
      cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);
      if (allocationProp.allocFlags.compressionType != CU_MEM_ALLOCATION_COMP_GENERIC) {
        errStream << "cuMemCreate was not able to allocate compressible memory";
        throw std::runtime_error(errStream.str());
      }

      result = cuMemMap(dptr, size, 0, allocationHandle, 0);
      if (result != CUDA_SUCCESS) {
        errStream << "could not map compressible memory. Error: " << result;
        throw std::runtime_error(errStream.str());
      }

      result = cuMemRelease(allocationHandle);
      if (result != CUDA_SUCCESS) {
        errStream << "could not release a memory handle representing a memory allocation. Error: " << result;
        throw std::runtime_error(errStream.str());
      }

      CUmemAccessDesc accessDescriptor;
      accessDescriptor.location.id = prop.location.id;
      accessDescriptor.location.type = prop.location.type;
      accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

      result = cuMemSetAccess(dptr, size, &accessDescriptor, 1);
      if (result != CUDA_SUCCESS) {
        errStream << "could not set the access flags. Error: " << result;
        throw std::runtime_error(errStream.str());
      }

      addr = reinterpret_cast<void*>(dptr);
      compressibleMemSizesTable[addr] = size;
    }
    catch (std::runtime_error& err) {
      std::stringstream stream;
      stream << "failed to allocate compressible memory because: " << err.what();
      throw std::runtime_error(stream.str());
    }
  }
  else {
    addr = this->allocGlobMem(size);
  }
  return addr;
}

void *ConcreteAPI::allocUnifiedMem(size_t size) {
  isFlagSet<DeviceSelected>(status);
  void *devPtr;
  cudaMallocManaged(&devPtr, size, cudaMemAttachGlobal);
  CHECK_ERR;
  if (allowedConcurrentManagedAccess) {
    cudaMemAdvise(devPtr, size, cudaMemAdviseSetPreferredLocation, currentDeviceId);
    CHECK_ERR;
  }
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

void ConcreteAPI::freeCompressibleMem(void *devPtr) {
  if (useCompressibleMemory) {
    isFlagSet<DeviceSelected>(status);
    try {
      if (devPtr != nullptr) {
        std::stringstream errStream;

        CUmemAllocationProp prop = {};
        cudaError_t err = setProp(&prop);

        if (err != cudaSuccess) {
          errStream << "could not set device properties";
          throw std::runtime_error(errStream.str());
        }

        auto size = compressibleMemSizesTable.at(devPtr);
        CUresult result = cuMemUnmap((CUdeviceptr)devPtr, size);
        if (result != CUDA_SUCCESS) {
          errStream << "could not unmap the backing memory of a given address range. Error: " << result;
          throw std::runtime_error(errStream.str());
        }

        result = cuMemAddressFree((CUdeviceptr)devPtr, size);
        if (result != CUDA_SUCCESS) {
          errStream << "could not free an address range reservation. Error: " << result;
          throw std::runtime_error(errStream.str());
        }
      }
    }
    catch (std::runtime_error& err) {
      std::stringstream stream;
      stream << "failed to deallocate compressible memory because: " << err.what();
      throw std::runtime_error(stream.str());
    }
  }
  else {
    return this->freeMem(devPtr);
  }
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
