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

ConcreteAPI::ConcreteAPI() {
  cuInit(0);
  CHECK_ERR;
  status[StatusID::DriverApiInitialized] = true;
}

void ConcreteAPI::setDevice(int deviceId) {
  currentDeviceId = deviceId;
  cudaSetDevice(currentDeviceId);
  CHECK_ERR;

  status[StatusID::DeviceSelected] = true;
}

void ConcreteAPI::initialize() {
  if (!status[StatusID::DeviceSelected]) {
    logError() << "Device has not been selected. Please, select device before calling initialize";
  }
  if (!status[StatusID::InterfaceInitialized]) {

    cudaStreamCreate(&defaultStream); CHECK_ERR;
    constexpr size_t concurrencyLevel = 32;
    circularStreamBuffer.resize(concurrencyLevel);
    for (auto &stream : circularStreamBuffer) {
      cudaStreamCreate(&stream);
      CHECK_ERR;
    }
    status[StatusID::InterfaceInitialized] = true;
  }
  else {
    logWarning() << "Device Interface has already been initialized";
  }
}

void ConcreteAPI::allocateStackMem() {
  isFlagSet<StatusID::DeviceSelected>(status);

  // try to detect the amount of temp. memory from the environment
  const size_t factor = 1024 * 1024 * 1024; //!< bytes in 1 GB

  try {
    char *valueString = std::getenv("DEVICE_STACK_MEM_SIZE");
    const auto id = currentDeviceId;
    if (!valueString) {
      logInfo(id)
          << "From device: env. variable \"DEVICE_STACK_MEM_SIZE\" has not been set. "
          << "The default amount of the device memory (1 GB) "
          << "is going to be used to store temp. variables during execution of compute-algorithms.";
    } else {
      double requestedStackMem = std::stod(std::string(valueString));
      maxStackMem = factor * requestedStackMem;
      logInfo(id) << "From device: env. variable \"DEVICE_STACK_MEM_SIZE\" has been detected. "
                  << requestedStackMem << "GB of the device memory is going to be used "
                  << "to store temp. variables during execution of compute-algorithms.";
    }
  } catch (const std::invalid_argument &err) {
    logError() << "DEVICE::ERROR: " << err.what() << ". File: " << __FILE__
               << ", line: " << __LINE__;
  } catch (const std::out_of_range &err) {
    logError() << "DEVICE::ERROR: " << err.what() << ". File: " << __FILE__
               << ", line: " << __LINE__;
  }

  cudaMalloc(&stackMemory, maxStackMem);
  CHECK_ERR;

  status[StatusID::StackMemAllocated] = true;
}

void ConcreteAPI::finalize() {
  if (status[StatusID::StackMemAllocated]) {
    cudaFree(stackMemory);
    CHECK_ERR;
    stackMemory = nullptr;
    stackMemByteCounter = 0;
    stackMemMeter = std::stack<size_t>{};
    status[StatusID::StackMemAllocated] = false;

  }
  if (status[StatusID::InterfaceInitialized]) {
    cudaStreamDestroy(defaultStream);
    for (auto &stream : circularStreamBuffer) {
      cudaStreamDestroy(stream);
      CHECK_ERR;
    }
    circularStreamBuffer.clear();
    status[StatusID::InterfaceInitialized] = false;
  }
}

int ConcreteAPI::getNumDevices() {
  int numDevices{};
  cudaGetDeviceCount(&numDevices);
  CHECK_ERR;
  return numDevices;
}

int ConcreteAPI::getDeviceId() {
  if (!status[StatusID::DeviceSelected]) {
    logError() << "Device has not been selected. Please, select device before requesting device Id";
  }
  return currentDeviceId;
}

size_t ConcreteAPI::getLaneSize() {
  return static_cast<size_t>(device::internals::WARP_SIZE);
}

unsigned ConcreteAPI::getMaxThreadBlockSize() {
  int blockSize{};
  cudaDeviceGetAttribute(&blockSize, cudaDevAttrMaxThreadsPerBlock, currentDeviceId);
  CHECK_ERR;
  return static_cast<unsigned>(blockSize);
}

unsigned ConcreteAPI::getMaxSharedMemSize() {
  int sharedMemSize{};
  cudaDeviceGetAttribute(&sharedMemSize, cudaDevAttrMaxSharedMemoryPerBlock, currentDeviceId);
  CHECK_ERR;
  return static_cast<unsigned>(sharedMemSize);
}

unsigned ConcreteAPI::getGlobMemAlignment() {
  // TODO: use cuDeviceGetAttribute
  return 128;
}

void ConcreteAPI::synchDevice() {
  isFlagSet<DeviceSelected>(status);
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
  isFlagSet<DeviceSelected>(status);
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
  isFlagSet<DeviceSelected>(status);
  nvtxRangePop();
#endif
}
