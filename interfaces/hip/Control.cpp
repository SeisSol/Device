#include "utils/logger.h"
#include <iostream>
#include <string>
#include <sstream>
#include <cuda.h>

#include "HipWrappedAPI.h"
#include "Internals.h"

using namespace device;

void ConcreteAPI::initialize() {
  hipInit(0); CHECK_ERR;
}

void ConcreteAPI::allocateStackMem() {

  // try to detect the amount of temp. memory from the environment
  const size_t Factor = 1024 * 1024 * 1024;  //!< bytes in 1 GB

  try {
    char *ValueString = std::getenv("DEVICE_STACK_MEM_SIZE");
    const auto id = m_CurrentDeviceId;
    if (!ValueString) {
      logInfo(id) << "From device: env. variable \"DEVICE_STACK_MEM_SIZE\" has not been set. "
                  << "The default amount of the device memory (1 GB) "
                  << "is going to be used to store temp. variables during execution of compute-algorithms.";
    }
    else {
      double RequestedStackMem = std::stod(std::string(ValueString));
      m_MaxStackMem = Factor * RequestedStackMem;
      logInfo(id) << "From device: env. variable \"DEVICE_STACK_MEM_SIZE\" has been detected. "
                  << RequestedStackMem << "GB of the device memory is going to be used "
                  << "to store temp. variables during execution of compute-algorithms.";
    }
  }
  catch (const std::invalid_argument &Err) {
    logError() << "DEVICE::ERROR: " << Err.what() << ". File: " << __FILE__ << ", line: " << __LINE__;
  }
  catch (const std::out_of_range& Err) {
    logError() << "DEVICE::ERROR: " << Err.what() << ". File: " << __FILE__ << ", line: " << __LINE__;
  }

  hipMalloc(&m_StackMemory, m_MaxStackMem); CHECK_ERR;

  constexpr size_t concurrencyLevel = 32;
  m_circularStreamBuffer.resize(concurrencyLevel);
  for (auto& stream: m_circularStreamBuffer) {
    hipStreamCreate(&stream); CHECK_ERR;
  }
};


void ConcreteAPI::finalize() {
  hipFree(m_StackMemory); CHECK_ERR;
  m_StackMemory = nullptr;
  for (auto& stream: m_circularStreamBuffer) {
    hipStreamDestroy(stream); CHECK_ERR;
  }
  m_circularStreamBuffer.clear();
  m_HasFinalized = true;
};


void ConcreteAPI::setDevice(int DeviceId) {
  m_CurrentDeviceId = DeviceId;
  hipSetDevice(m_CurrentDeviceId); CHECK_ERR;
}


int ConcreteAPI::getNumDevices() {
  int numDevices{};
  hipGetDeviceCount(&numDevices); CHECK_ERR;
  return numDevices;
}


unsigned ConcreteAPI::getMaxThreadBlockSize() {
  int BlockSize{};
  hipDeviceGetAttribute(&BlockSize, hipDeviceAttributeMaxThreadsPerBlock, m_CurrentDeviceId); CHECK_ERR;
  return static_cast<unsigned>(BlockSize);
}


unsigned ConcreteAPI::getMaxSharedMemSize() {
  int SharedMemSize{};
  hipDeviceGetAttribute(&SharedMemSize, hipDeviceAttributeMaxSharedMemoryPerBlock, m_CurrentDeviceId); CHECK_ERR;
  return static_cast<unsigned>(SharedMemSize);
}


unsigned ConcreteAPI::getGlobMemAlignment() {
  // TODO: use hipDeviceGetAttribute
  return 128;
}


void ConcreteAPI::synchDevice() {
  hipDeviceSynchronize(); CHECK_ERR;
}


std::string ConcreteAPI::getDeviceInfoAsText(int DeviceId) {
  hipDeviceProp_t Property;
  hipGetDeviceProperties(&Property, DeviceId); CHECK_ERR;

  std::ostringstream Info;
  Info << "Name: " << Property.name << '\n';
  Info << "totalGlobalMem: " << Property.totalGlobalMem << '\n';
  Info << "sharedMemPerBlock: " << Property.sharedMemPerBlock << '\n';
  Info << "regsPerBlock: " << Property.regsPerBlock << '\n';
  Info << "warpSize: " << Property.warpSize << '\n';
  Info << "maxThreadsPerBlock: " << Property.maxThreadsPerBlock << '\n';
  Info << "totalConstMem: " << Property.totalConstMem << '\n';
  Info << "clockRate: " << Property.clockRate << '\n';
  Info << "multiProcessorCount: " << Property.multiProcessorCount << '\n';
  Info << "canMapHostMemory: " << Property.canMapHostMemory << '\n';
  Info << "computeMode: " << Property.computeMode << '\n';
  Info << "concurrentKernels: " << Property.concurrentKernels << '\n';
  Info << "pciBusID: " << Property.pciBusID << '\n';
  Info << "pciDeviceID: " << Property.pciDeviceID << '\n';

  return Info.str();
}

/**
 * No implementation, since there is no HIP alternative for nvToolsExt
 */
void ConcreteAPI::putProfilingMark(const std::string &Name, ProfilingColors Color) {
}

/**
 * No implementation, since there is no HIP alternative for nvToolsExt
 */
void ConcreteAPI::popLastProfilingMark() {

}
