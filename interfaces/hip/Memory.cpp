#include "hip/hip_runtime.h"
#include <assert.h>
#include <sstream>
#include <iostream>

#include "HipWrappedAPI.h"
#include "Internals.h"

using namespace device;

void* ConcreteAPI::allocGlobMem(size_t Size) {
  void *DevPtr;
  hipMalloc(&DevPtr, Size); CHECK_ERR;
  m_Statistics.AllocatedMemBytes += Size;
  m_MemToSizeMap[DevPtr] = Size;
  return DevPtr;
}


void* ConcreteAPI::allocUnifiedMem(size_t Size) {
  void *DevPtr;
  hipMallocManaged(&DevPtr, Size, hipMemAttachGlobal); CHECK_ERR;
  m_Statistics.AllocatedMemBytes += Size;
  m_Statistics.AllocatedUnifiedMemBytes += Size;
  m_MemToSizeMap[DevPtr] = Size;
  return DevPtr;
}


void* ConcreteAPI::allocPinnedMem(size_t Size) {
  void *DevPtr;
  hipHostMalloc(&DevPtr, Size); CHECK_ERR;
  m_Statistics.AllocatedMemBytes += Size;
  m_MemToSizeMap[DevPtr] = Size;
  return DevPtr;
}


void ConcreteAPI::freeMem(void *DevPtr) {
  assert((m_MemToSizeMap.find(DevPtr) != m_MemToSizeMap.end())
         && "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  m_Statistics.DeallocatedMemBytes += m_MemToSizeMap[DevPtr];
  hipFree(DevPtr); CHECK_ERR;
}


void ConcreteAPI::freePinnedMem(void *DevPtr) {
  assert((m_MemToSizeMap.find(DevPtr) != m_MemToSizeMap.end())
         && "DEVICE: an attempt to delete mem. which has not been allocated. unknown pointer");
  m_Statistics.DeallocatedMemBytes += m_MemToSizeMap[DevPtr];
  hipHostFree(DevPtr); CHECK_ERR;
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

size_t ConcreteAPI::getMaxAvailableMem() {
  hipDeviceProp_t Property;
  hipGetDeviceProperties(&Property, m_CurrentDeviceId); CHECK_ERR;
  return Property.totalGlobalMem;
}

size_t ConcreteAPI::getCurrentlyOccupiedMem() {
  return m_Statistics.AllocatedMemBytes;
}

size_t ConcreteAPI::getCurrentlyOccupiedUnifiedMem() {
  return m_Statistics.AllocatedUnifiedMemBytes;
}