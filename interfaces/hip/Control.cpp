// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "utils/logger.h"
#include "utils/env.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include "hip/hip_runtime.h"

#ifdef PROFILING_ENABLED
#include "roctx.h"
#endif

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

  hipDeviceProp_t properties{};
  hipGetDeviceProperties(&properties, currentDeviceId);
  CHECK_ERR;

  // NOTE: hipDeviceGetAttribute internally calls hipGetDeviceProperties; hence it doesn't make sense to use it here

  if constexpr (HIP_VERSION >= 60200000) {
    // cf. https://rocm.docs.amd.com/en/docs-6.2.0/about/release-notes.html
    // (before 6.2.0, the flag hipDeviceAttributePageableMemoryAccessUsesHostPageTables had effectively the same effect)
    // (cf. https://github.com/ROCm/clr/commit/7d5b4a8f7a7d34f008d65277f8aae4c98a6da375#diff-596cd550f7fdef76b39f1b7b179b20128313dd9cc9ec662b2eae562efa2b7f33L405 )
    usmDefault = properties.integrated != 0;
  }
  else {
    usmDefault = properties.directManagedMemAccessFromHost != 0 && properties.pageableMemoryAccessUsesHostPageTables != 0;
  }

  hipDeviceGetStreamPriorityRange(&priorityMin, &priorityMax);
  CHECK_ERR;

  status[StatusID::DeviceSelected] = true;
}

bool ConcreteAPI::isUnifiedMemoryDefault() {
  return usmDefault;
}

void ConcreteAPI::initialize() {
  if (!status[StatusID::DeviceSelected]) {
    logError() << "Device has not been selected. Please, select device before calling initialize";
  }

  if (!status[StatusID::InterfaceInitialized]) {
    status[StatusID::InterfaceInitialized] = true;
    hipStreamCreateWithFlags(&defaultStream, hipStreamNonBlocking); CHECK_ERR;
    hipEventCreate(&defaultStreamEvent); CHECK_ERR;
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
    if (!valueString) {
      printer.printInfo()
          << "From device: env. variable \"DEVICE_STACK_MEM_SIZE\" has not been set. "
          << "The default amount of the device memory (1 GB) "
          << "is going to be used to store temp. variables during execution of compute-algorithms.";
    } else {
      double requestedStackMem = std::stod(std::string(valueString));
      maxStackMem = factor * requestedStackMem;
      printer.printInfo() << "From device: env. variable \"DEVICE_STACK_MEM_SIZE\" has been detected. "
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

  if (status[StatusID::InterfaceInitialized]) {
    hipStreamDestroy(defaultStream); CHECK_ERR;
    hipEventDestroy(defaultStreamEvent); CHECK_ERR;
    if (!genericStreams.empty()) {
      printer.printInfo() << "DEVICE::WARNING:" << genericStreams.size()
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

std::string ConcreteAPI::getApiName() {
  return "HIP";
}

std::string ConcreteAPI::getDeviceName(int deviceId) {
  hipDeviceProp_t property;
  hipGetDeviceProperties(&property, deviceId);
  CHECK_ERR;

  return property.name;
}

std::string ConcreteAPI::getPciAddress(int deviceId) {
  hipDeviceProp_t property;
  hipGetDeviceProperties(&property, deviceId);
  CHECK_ERR;

  std::ostringstream str;
  str << std::setfill('0') << std::setw(4) << std::hex << property.pciDomainID << ":" << std::setw(2) << property.pciBusID << ":" << property.pciDeviceID << "." << "0";
  return str.str();
}

void ConcreteAPI::putProfilingMark(const std::string &name, ProfilingColors color) {
  // colors are not yet supported here
#ifdef PROFILING_ENABLED
  isFlagSet<DeviceSelected>(status);
  roctxRangePush(name.c_str());
#endif
}

void ConcreteAPI::popLastProfilingMark() {
#ifdef PROFILING_ENABLED
  isFlagSet<DeviceSelected>(status);
  roctxRangePop();
#endif
}

void ConcreteAPI::setupPrinting(int rank) {
  printer.setRank(rank);
}

