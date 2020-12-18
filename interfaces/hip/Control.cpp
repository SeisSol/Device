#include "utils/logger.h"
#include <iostream>
#include <string>
#include <sstream>
#include "hip/hip_runtime.h"

#include "HipWrappedAPI.h"
#include "Internals.h"

using namespace device;

void ConcreteAPI::initialize() {
  hipInit(0); CHECK_ERR;
}

void ConcreteAPI::allocateStackMem() {

  // try to detect the amount of temp. memory from the environment
  const size_t factor = 1024 * 1024 * 1024;  //!< bytes in 1 GB

  try {
    char *valueString = std::getenv("DEVICE_STACK_MEM_SIZE");
    const auto id = m_currentDeviceId;
    if (!valueString) {
      logInfo(id) << "From device: env. variable \"DEVICE_STACK_MEM_SIZE\" has not been set. "
                  << "The default amount of the device memory (1 GB) "
                  << "is going to be used to store temp. variables during execution of compute-algorithms.";
    }
    else {
      double requestedStackMem = std::stod(std::string(valueString));
      m_maxStackMem = factor * requestedStackMem;
      logInfo(id) << "From device: env. variable \"DEVICE_STACK_MEM_SIZE\" has been detected. "
                  << requestedStackMem << "GB of the device memory is going to be used "
                  << "to store temp. variables during execution of compute-algorithms.";
    }
  }
  catch (const std::invalid_argument &err) {
    logError() << "DEVICE::ERROR: " << err.what() << ". File: " << __FILE__ << ", line: " << __LINE__;
  }
  catch (const std::out_of_range& err) {
    logError() << "DEVICE::ERROR: " << err.what() << ". File: " << __FILE__ << ", line: " << __LINE__;
  }

  hipMalloc(&m_stackMemory, m_maxStackMem); CHECK_ERR;

  constexpr size_t concurrencyLevel = 32;
  m_circularStreamBuffer.resize(concurrencyLevel);
  for (auto& stream: m_circularStreamBuffer) {
    hipStreamCreate(&stream); CHECK_ERR;
  }
};


void ConcreteAPI::finalize() {
  hipFree(m_stackMemory); CHECK_ERR;
  m_stackMemory = nullptr;
  for (auto& stream: m_circularStreamBuffer) {
    hipStreamDestroy(stream); CHECK_ERR;
  }
  m_circularStreamBuffer.clear();
  m_isFinalized = true;
};


void ConcreteAPI::setDevice(int deviceId) {
  m_currentDeviceId = deviceId;
  hipSetDevice(m_currentDeviceId); CHECK_ERR;
}


int ConcreteAPI::getNumDevices() {
  int numDevices{};
  hipGetDeviceCount(&numDevices); CHECK_ERR;
  return numDevices;
}


unsigned ConcreteAPI::getMaxThreadBlockSize() {
  int blockSize{};
  hipDeviceGetAttribute(&blockSize, hipDeviceAttributeMaxThreadsPerBlock, m_currentDeviceId); CHECK_ERR;
  return static_cast<unsigned>(blockSize);
}


unsigned ConcreteAPI::getMaxSharedMemSize() {
  int sharedMemSize{};
  hipDeviceGetAttribute(&sharedMemSize, hipDeviceAttributeMaxSharedMemoryPerBlock, m_currentDeviceId); CHECK_ERR;
  return static_cast<unsigned>(sharedMemSize);
}


unsigned ConcreteAPI::getGlobMemAlignment() {
  // TODO: use hipDeviceGetAttribute
  return 128;
}


void ConcreteAPI::synchDevice() {
  hipDeviceSynchronize(); CHECK_ERR;
}


std::string ConcreteAPI::getDeviceInfoAsText(int deviceId) {
  hipDeviceProp_t property;
  hipGetDeviceProperties(&property, deviceId); CHECK_ERR;

  std::ostringstream info;
  info << "Name: " << property.name << '\n';
  info << "totalGlobalMem: " << property.totalGlobalMem << '\n';
  info << "sharedMemPerBlock: " << property.sharedMemPerBlock << '\n';
  info << "regsPerBlock: " << property.regsPerBlock << '\n';
  info << "warpSize: " << property.warpSize << '\n';
  info << "maxThreadsPerBlock: " << property.maxThreadsPerBlock << '\n';
  info << "totalConstMem: " << property.totalConstMem << '\n';
  info << "clockRate: " << property.clockRate << '\n';
  info << "multiProcessorCount: " << property.multiProcessorCount << '\n';
  info << "canMapHostMemory: " << property.canMapHostMemory << '\n';
  info << "computeMode: " << property.computeMode << '\n';
  info << "concurrentKernels: " << property.concurrentKernels << '\n';
  info << "pciBusID: " << property.pciBusID << '\n';
  info << "pciDeviceID: " << property.pciDeviceID << '\n';

  return info.str();
}

/**
 * No implementation, since there is no HIP alternative for nvToolsExt
 */
void ConcreteAPI::putProfilingMark(const std::string &name, ProfilingColors color) {
}

/**
 * No implementation, since there is no HIP alternative for nvToolsExt
 */
void ConcreteAPI::popLastProfilingMark() {

}
