#include "utils/logger.h"
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <string>

#ifdef PROFILING_ENABLED
#include <nvToolsExt.h>
#endif

#include "CudaWrappedAPI.h"
#include "Internals.h"

using namespace device;

void ConcreteAPI::initialize() {
  cuInit(0);
  CHECK_ERR;
}

void ConcreteAPI::allocateStackMem() {

  // try to detect the amount of temp. memory from the environment
  const size_t factor = 1024 * 1024 * 1024; //!< bytes in 1 GB

  try {
    char *valueString = std::getenv("DEVICE_STACK_MEM_SIZE");
    const auto id = m_currentDeviceId;
    if (!valueString) {
      logInfo(id)
          << "From device: env. variable \"DEVICE_STACK_MEM_SIZE\" has not been set. "
          << "The default amount of the device memory (1 GB) "
          << "is going to be used to store temp. variables during execution of compute-algorithms.";
    } else {
      double RequestedStackMem = std::stod(std::string(valueString));
      m_maxStackMem = factor * RequestedStackMem;
      logInfo(id) << "From device: env. variable \"DEVICE_STACK_MEM_SIZE\" has been detected. "
                  << RequestedStackMem << "GB of the device memory is going to be used "
                  << "to store temp. variables during execution of compute-algorithms.";
    }
  } catch (const std::invalid_argument &err) {
    logError() << "DEVICE::ERROR: " << err.what() << ". File: " << __FILE__
               << ", line: " << __LINE__;
  } catch (const std::out_of_range &err) {
    logError() << "DEVICE::ERROR: " << err.what() << ". File: " << __FILE__
               << ", line: " << __LINE__;
  }

  cudaMalloc(&m_stackMemory, m_maxStackMem);
  CHECK_ERR;

  constexpr size_t concurrencyLevel = 32;
  m_circularStreamBuffer.resize(concurrencyLevel);
  for (auto &stream : m_circularStreamBuffer) {
    cudaStreamCreate(&stream);
    CHECK_ERR;
  }
};

void ConcreteAPI::finalize() {
  cudaFree(m_stackMemory);
  CHECK_ERR;
  m_stackMemory = nullptr;
  for (auto &stream : m_circularStreamBuffer) {
    cudaStreamDestroy(stream);
    CHECK_ERR;
  }
  m_circularStreamBuffer.clear();
  m_isFinalized = true;
};

void ConcreteAPI::setDevice(int deviceId) {
  m_currentDeviceId = deviceId;
  cudaSetDevice(m_currentDeviceId);
  CHECK_ERR;
}

int ConcreteAPI::getNumDevices() {
  int numDevices{};
  cudaGetDeviceCount(&numDevices);
  CHECK_ERR;
  return numDevices;
}

unsigned ConcreteAPI::getMaxThreadBlockSize() {
  int blockSize{};
  cudaDeviceGetAttribute(&blockSize, cudaDevAttrMaxThreadsPerBlock, m_currentDeviceId);
  CHECK_ERR;
  return static_cast<unsigned>(blockSize);
}

unsigned ConcreteAPI::getMaxSharedMemSize() {
  int sharedMemSize{};
  cudaDeviceGetAttribute(&sharedMemSize, cudaDevAttrMaxSharedMemoryPerBlock, m_currentDeviceId);
  CHECK_ERR;
  return static_cast<unsigned>(sharedMemSize);
}

unsigned ConcreteAPI::getGlobMemAlignment() {
  // TODO: use cuDeviceGetAttribute
  return 128;
}

void ConcreteAPI::synchDevice() {
  cudaDeviceSynchronize();
  CHECK_ERR;
}

std::string ConcreteAPI::getDeviceInfoAsText(int deviceId) {
  cudaDeviceProp property;
  cudaGetDeviceProperties(&property, deviceId);
  CHECK_ERR;

  std::ostringstream info;
  info << "Name: " << property.name << '\n';
  info << "totalGlobalMem: " << property.totalGlobalMem << '\n';
  info << "sharedMemPerBlock: " << property.sharedMemPerBlock << '\n';
  info << "regsPerBlock: " << property.regsPerBlock << '\n';
  info << "warpSize: " << property.warpSize << '\n';
  info << "memPitch: " << property.memPitch << '\n';
  info << "maxThreadsPerBlock: " << property.maxThreadsPerBlock << '\n';
  info << "totalConstMem: " << property.totalConstMem << '\n';
  info << "clockRate: " << property.clockRate << '\n';
  info << "multiProcessorCount: " << property.multiProcessorCount << '\n';
  info << "integrated: " << property.integrated << '\n';
  info << "canMapHostMemory: " << property.canMapHostMemory << '\n';
  info << "computeMode: " << property.computeMode << '\n';
  info << "concurrentKernels: " << property.concurrentKernels << '\n';
  info << "pciBusID: " << property.pciBusID << '\n';
  info << "pciDeviceID: " << property.pciDeviceID << '\n';

  return info.str();
}

void ConcreteAPI::putProfilingMark(const std::string &name, ProfilingColors color) {
#ifdef PROFILING_ENABLED
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = static_cast<uint32_t>(color);
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = name.c_str();
  nvtxRangePushEx(&eventAttrib);
#endif
}

void ConcreteAPI::popLastProfilingMark() {
#ifdef PROFILING_ENABLED
  nvtxRangePop();
#endif
}
