#include "utils/logger.h"
#include <iostream>
#include <sstream>
#include <string>
#include "hip/hip_runtime.h"

#include "HipWrappedAPI.h"
#include "Internals.h"

using namespace device;

ConcreteAPI::ConcreteAPI() {
  hipInit(0);
  CHECK_ERR;
  status[StatusID::DriverApiInitialized] = true;
}

void ConcreteAPI::setDevice(int deviceId) {
  currentDeviceId = deviceId;
  hipSetDevice(currentDeviceId);
  CHECK_ERR;

  // Note: the following sets the initial HIP context
  hipFree(nullptr);
  CHECK_ERR;

  status[StatusID::DeviceSelected] = true;
}

void ConcreteAPI::initialize() {
  if (!status[StatusID::DeviceSelected]) {
    logError() << "Device has not been selected. Please, select device before calling initialize";
  }

  if (!status[StatusID::InterfaceInitialized]) {
    status[StatusID::InterfaceInitialized] = true;
    hipStreamCreateWithFlags(&defaultStream, hipStreamNonBlocking); CHECK_ERR;
    hipEventCreate(&defaultStreamEvent); CHECK_ERR;

    this->createCircularStreamAndEvents();
  }
  else {
    logWarning() << "Device Interface has already been initialized";
  }
}

void ConcreteAPI::createCircularStreamAndEvents() {
  isFlagSet<StatusID::InterfaceInitialized>(status);

#ifdef CUDA_UNDERHOOD
  constexpr size_t concurrencyLevel{32};
#else
  constexpr size_t concurrencyLevel{8};
#endif
  circularStreamBuffer.resize(concurrencyLevel);
  for (auto &stream : circularStreamBuffer) {
    hipStreamCreateWithFlags(&stream, hipStreamNonBlocking); CHECK_ERR;
    CHECK_ERR;
  }

  circularStreamEvents.resize(concurrencyLevel);
  for (auto &event : circularStreamEvents) {
    hipEventCreate(&event);
    CHECK_ERR;
  }

  status[StatusID::CircularStreamBufferInitialized] = true;
}

void ConcreteAPI::allocateStackMem() {
  isFlagSet<StatusID::DeviceSelected>(status);

  // try to detect the amount of temp. memory from the environment
  const size_t factor = 1024 * 1024 * 1024; //!< bytes in 1 GB

  try {
    char *valueString = std::getenv("DEVICE_STACK_MEM_SIZE");
    const auto rank = getMpiRankFromEnv();
    if (!valueString) {
      logInfo(rank)
          << "From device: env. variable \"DEVICE_STACK_MEM_SIZE\" has not been set. "
          << "The default amount of the device memory (1 GB) "
          << "is going to be used to store temp. variables during execution of compute-algorithms.";
    } else {
      double requestedStackMem = std::stod(std::string(valueString));
      maxStackMem = factor * requestedStackMem;
      logInfo(rank) << "From device: env. variable \"DEVICE_STACK_MEM_SIZE\" has been detected. "
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

  hipMalloc(&stackMemory, maxStackMem);
  CHECK_ERR;

  status[StatusID::StackMemAllocated] = true;
}

void ConcreteAPI::finalize() {
  if (status[StatusID::StackMemAllocated]) {
    hipFree(stackMemory);
    CHECK_ERR;
    stackMemory = nullptr;
    stackMemByteCounter = 0;
    stackMemMeter = std::stack<size_t>{};
    status[StatusID::StackMemAllocated] = false;
  }

  if (status[StatusID::CircularStreamBufferInitialized]) {
    for (auto &stream : circularStreamBuffer) {
      hipStreamDestroy(stream);
      CHECK_ERR;
    }
    circularStreamBuffer.clear();

    for (auto &event : circularStreamEvents) {
      hipEventDestroy(event);
      CHECK_ERR;
    }
    circularStreamEvents.clear();

    for (auto &graphInstance : graphs) {
      hipGraphExecDestroy(graphInstance.instance);
      CHECK_ERR;

      hipGraphDestroy(graphInstance.graph);
      CHECK_ERR;

      hipStreamDestroy(graphInstance.graphExecutionStream);
      CHECK_ERR;

      hipEventDestroy(graphInstance.graphCaptureEvent);
      CHECK_ERR;
    }
    graphs.clear();

    status[StatusID::CircularStreamBufferInitialized] = false;
  }

  if (status[StatusID::InterfaceInitialized]) {
    hipStreamDestroy(defaultStream); CHECK_ERR;
    hipEventDestroy(defaultStreamEvent); CHECK_ERR;
    if (!genericStreams.empty()) {
      logInfo(currentDeviceId) << "DEVICE::WARNING:" << genericStreams.size()
                               << "device generic stream(s) were not deleted.";
      for (auto stream : genericStreams) {
        hipStreamDestroy(stream); CHECK_ERR;
      }
    }
    status[StatusID::InterfaceInitialized] = false;
  }
}


int ConcreteAPI::getNumDevices() {
  int numDevices{};
  hipGetDeviceCount(&numDevices);
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
  hipDeviceGetAttribute(&blockSize, hipDeviceAttributeMaxThreadsPerBlock, currentDeviceId); CHECK_ERR;
  CHECK_ERR;
  return static_cast<unsigned>(blockSize);
}

unsigned ConcreteAPI::getMaxSharedMemSize() {
  int sharedMemSize{};
  hipDeviceGetAttribute(&sharedMemSize, hipDeviceAttributeMaxSharedMemoryPerBlock, currentDeviceId); CHECK_ERR;
  CHECK_ERR;
  return static_cast<unsigned>(sharedMemSize);
}

unsigned ConcreteAPI::getGlobMemAlignment() {
  // TODO: use hipDeviceGetAttribute
#ifdef CUDA_UNDERHOOD
  return 128;
#else
  return 256;
#endif
}

void ConcreteAPI::syncDevice() {
  isFlagSet<DeviceSelected>(status);
  hipDeviceSynchronize();
  CHECK_ERR;
}

std::string ConcreteAPI::getDeviceInfoAsText(int deviceId) {
  hipDeviceProp_t property;
  hipGetDeviceProperties(&property, deviceId);
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
