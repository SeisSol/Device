// SPDX-FileCopyrightText: 2021 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Internals.h"
#include "SyclWrappedAPI.h"

#include <iostream>

#ifdef SYCL_BACKEND_ZE
#include <ze_api.h>
#endif

// should only exist if <ze_api.h> has been included
#ifdef ZE_MEMORY_COMPRESSION_HINTS_EXT_NAME

namespace {
constexpr bool HasCompress = true;

void* allocCompressed(sycl::queue& queue, size_t size) {
  void* ptrOut{nullptr};
  void** ptrptr = &ptrOut;

  queue.host_task([=](const sycl::interop_handle& handle) {
    // for the basic code structure cf.
    // https://github.com/intel/llvm/blob/c757480800a9e0224b81e509bf88d2b77067da69/unified-runtime/source/adapters/level_zero/usm.cpp#L183-L204

    ze_memory_compression_hints_ext_desc_t compresshints{};
    compresshints.stype = ZE_STRUCTURE_TYPE_MEMORY_COMPRESSION_HINTS_EXT_DESC;
    compresshints.pNext = nullptr;
    compresshints.flags = ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_COMPRESSED;
    ze_device_mem_alloc_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    desc.pNext = &compresshints;
    desc.flags = 0;
    desc.ordinal = 0;

    const ze_context_handle_t nativeContext = handle.get_native_context<sycl::backend::ze>();
    const ze_device_handle_t nativeDevice = handle.get_native_device<sycl::backend::ze>();

    zeMemAllocDevice(nativeContext, &desc, size, 128, nativeDevice, ptrptr);
  });

  // wait for the pointer on the stack to be written out
  waitCheck(queue);

  return ptrOut;
}

} // namespace

#else

namespace {
constexpr bool HasCompress = false;

void* allocCompressed(sycl::queue& queue, size_t size) { return nullptr; }

} // namespace

#endif

using namespace device;
using namespace device::internals;

void* ConcreteAPI::allocGlobMem(size_t size, bool compress) {
  void* ptr{nullptr};
  if (HasCompress && compress) {
    ptr = allocCompressed(this->currentDefaultQueue(), size);
  } else {
    ptr = malloc_device(size, this->currentDefaultQueue());
    waitCheck(this->currentDefaultQueue());
  }
  this->currentStatistics().allocatedMemBytes += size;
  this->currentMemoryToSizeMap().insert({ptr, size});
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
  if (devPtr != nullptr) {
    if (this->currentMemoryToSizeMap().find(devPtr) == this->currentMemoryToSizeMap().end()) {
      throw std::invalid_argument(
          this->getDeviceInfoAsText(getDeviceId())
              .append("an attempt to delete memory that has not been allocated. Is this "
                      "a pointer to this device or was this a double free?"));
    }

    this->currentStatistics().deallocatedMemBytes += this->currentMemoryToSizeMap().at(devPtr);
    this->currentMemoryToSizeMap().erase(devPtr);
    free(devPtr, this->currentDefaultQueue().get_context());
    waitCheck(currentDefaultQueue());
  }
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
