#include <assert.h>
#include <sstream>
#include <iostream>

#include "CudaWrappedAPI.h"
#include "Internals.h"

using namespace device;

void* ConcreteAPI::allocGlobMem(size_t Size) {
  void *DevPtr;
  cudaMalloc(&DevPtr, Size); CHECK_ERR;
  m_Statistics.AllocatedMemBytes += Size;
  m_MemToSizeMap[DevPtr] = Size;
  return DevPtr;
}


void* ConcreteAPI::allocUnifiedMem(size_t Size) {
  void *DevPtr;
  cudaMallocManaged(&DevPtr, Size, cudaMemAttachGlobal); CHECK_ERR;
  m_Statistics.AllocatedMemBytes += Size;
  m_Statistics.AllocatedUnifiedMemBytes += Size;
  m_MemToSizeMap[DevPtr] = Size;
  return DevPtr;
}


void* ConcreteAPI::allocPinnedMem(size_t Size) {
  void *DevPtr;
  cudaMallocHost(&DevPtr, Size); CHECK_ERR;
  m_Statistics.AllocatedMemBytes += Size;
  m_MemToSizeMap[DevPtr] = Size;
  return DevPtr;
}


void ConcreteAPI::freeMem(void *DevPtr) {
  assert((m_MemToSizeMap.find(DevPtr) != m_MemToSizeMap.end())
          && "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  m_Statistics.DeallocatedMemBytes += m_MemToSizeMap[DevPtr];
  cudaFree(DevPtr); CHECK_ERR;
}


void ConcreteAPI::freePinnedMem(void *DevPtr) {
  assert((m_MemToSizeMap.find(DevPtr) != m_MemToSizeMap.end())
         && "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  m_Statistics.DeallocatedMemBytes += m_MemToSizeMap[DevPtr];
  cudaFreeHost(DevPtr); CHECK_ERR;
}


char* ConcreteAPI::getStackMemory(size_t RequestedBytes) {
  assert(((m_StackMemByteCounter + RequestedBytes) < m_MaxStackMem) && "DEVICE:: run out of a device stack memory");
  char *Mem = &m_StackMemory[m_StackMemByteCounter];
  m_StackMemByteCounter += RequestedBytes;
  m_StackMemMeter.push(RequestedBytes);
  return Mem;
}


void ConcreteAPI::popStackMemory() {
  m_StackMemByteCounter -= m_StackMemMeter.top();
  m_StackMemMeter.pop();
}

std::string ConcreteAPI::getMemLeaksReport() {
  std::ostringstream Report{};
  Report << "Memory Leaks, bytes: " << (m_Statistics.AllocatedMemBytes - m_Statistics.DeallocatedMemBytes) << '\n';
  Report << "Stack Memory Leaks, bytes: " << m_StackMemByteCounter << '\n';
  return Report.str();
}


__global__ void kernel_touchMemory(real *Ptr, size_t Size, bool Clean) {
  int Id = threadIdx.x + blockIdx.x * blockDim.x;
  if (Id < Size) {
    if (Clean) {
      Ptr[Id] = 0.0;
    }
    else {
      real Value = Ptr[Id];
      // Do something dummy here. We just need to check the pointers point to valid memory locations.
      // Avoid compiler optimization. Possibly, implement a dummy code with asm.
      Value += 1.0;
    }
  }
}

void ConcreteAPI::touchMemory(real *Ptr, size_t Size, bool Clean) {
  dim3 Block(256, 1, 1);
  dim3 Grid = internals::computeGrid1D(Block, Size);
  kernel_touchMemory<<<Grid, Block>>>(Ptr, Size, Clean); CHECK_ERR;
}


__global__ void kernel_touchBatchedMemory(real **BasePtr, unsigned ElementSize, bool Clean) {
  real *Element = BasePtr[blockIdx.x];
  int Id = threadIdx.x;
  while (Id < ElementSize) {
    if (Clean) {
      Element[Id] = 0.0;
    } else {
      real Value = Element[Id];
      // Do something dummy here. We just need to check the pointers point to valid memory locations.
      // Avoid compiler optimization. Possibly, implement a dummy code with asm.
      Value += 1.0;
    }
    Id += blockDim.x;
  }
}

void ConcreteAPI::touchBatchedMemory(real **BasePtr, unsigned ElementSize, unsigned NumElements, bool Clean) {
  dim3 Block(256, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_touchBatchedMemory<<<Grid, Block>>>(BasePtr, ElementSize, Clean); CHECK_ERR;
}

size_t ConcreteAPI::getMaxAvailableMem() {
  cudaDeviceProp Property;
  cudaGetDeviceProperties(&Property, m_CurrentDeviceId); CHECK_ERR;
  return Property.totalGlobalMem;
}

size_t ConcreteAPI::getCurrentlyOccupiedMem() {
  return m_Statistics.AllocatedMemBytes;
}

size_t ConcreteAPI::getCurrentlyOccupiedUnifiedMem() {
  return m_Statistics.AllocatedUnifiedMemBytes;
}