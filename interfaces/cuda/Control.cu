// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "utils/logger.h"
#include "utils/env.h"
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#ifdef PROFILING_ENABLED
#include <nvToolsExt.h>
#endif

#include "CudaWrappedAPI.h"
#include "Internals.h"

namespace {
// `static` is a bit out of place here; but we treat the whole class as an effective singleton anyways

#ifdef DEVICE_CONTEXT_GLOBAL
int currentDeviceId = 0;
#else
thread_local int currentDeviceId = 0;
#endif
}

using namespace device;

ConcreteAPI::ConcreteAPI() {
  cuInit(0);
  CHECK_ERR;
  status[StatusID::DriverApiInitialized] = true;
}

void ConcreteAPI::setDevice(int deviceId) {
  
  currentDeviceId = deviceId;

  cudaSetDevice(deviceId);
  CHECK_ERR;

  // Note: the following sets the initial CUDA context
  cudaFree(nullptr);
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
    cudaStreamCreateWithFlags(&defaultStream, cudaStreamNonBlocking); CHECK_ERR;

    int result{0};
    cudaDeviceGetAttribute(&result, cudaDevAttrConcurrentManagedAccess, getDeviceId());
    CHECK_ERR;
    allowedConcurrentManagedAccess = result != 0;

    cudaDeviceGetAttribute(&result, cudaDevAttrDirectManagedMemAccessFromHost, getDeviceId());
    usmDefault = result != 0;

    cudaDeviceGetStreamPriorityRange(&priorityMin, &priorityMax);
    CHECK_ERR;

    int canCompressProto = 0;
    cuDeviceGetAttribute(&canCompressProto, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, getDeviceId());
    canCompress = canCompressProto != 0;
  }
  else {
    logWarning() << "Device Interface has already been initialized";
  }
}

void ConcreteAPI::finalize() {
  if (status[StatusID::InterfaceInitialized]) {
    cudaStreamDestroy(defaultStream); CHECK_ERR;
    if (!genericStreams.empty()) {
      logInfo() << "DEVICE::WARNING:" << genericStreams.size()
                               << "device generic stream(s) were not deleted.";
      for (auto stream : genericStreams) {
        cudaStreamDestroy(stream); CHECK_ERR;
      }
    }
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

unsigned ConcreteAPI::getGlobMemAlignment() {
  // TODO: use cuDeviceGetAttribute
  return 128;
}

void ConcreteAPI::syncDevice() {
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
  info << "multiProcessorCount: " << property.multiProcessorCount << '\n';
  info << "integrated: " << property.integrated << '\n';
  info << "canMapHostMemory: " << property.canMapHostMemory << '\n';
  info << "concurrentKernels: " << property.concurrentKernels << '\n';
  info << "pciBusID: " << property.pciBusID << '\n';
  info << "pciDeviceID: " << property.pciDeviceID << '\n';

  return info.str();
}

std::string ConcreteAPI::getApiName() {
  return "CUDA";
}

std::string ConcreteAPI::getDeviceName(int deviceId) {
  cudaDeviceProp property;
  cudaGetDeviceProperties(&property, deviceId);
  CHECK_ERR;

  return property.name;
}

std::string ConcreteAPI::getPciAddress(int deviceId) {
  cudaDeviceProp property;
  cudaGetDeviceProperties(&property, deviceId);
  CHECK_ERR;

  std::ostringstream str;
  str << std::setfill('0') << std::setw(4) << std::hex << property.pciDomainID << ":" << std::setw(2) << property.pciBusID << ":" << property.pciDeviceID << "." << "0";
  return str.str();
}

void ConcreteAPI::profilingMessage(const std::string& message) {
#ifdef PROFILING_ENABLED
  isFlagSet<DeviceSelected>(status);
  nvtxMark(message.c_str());
#endif
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

void ConcreteAPI::setupPrinting(int rank) {

}

