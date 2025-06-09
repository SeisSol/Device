// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "device.h"
#include "Algorithms.h"

#ifdef DEVICE_LANG_CUDA
#include "interfaces/cuda/CudaWrappedAPI.h"
#define DEVICE_ENABLED
#elif DEVICE_LANG_HIP
#include "interfaces/hip/HipWrappedAPI.h"
#define DEVICE_ENABLED
#elif DEVICE_LANG_SYCL
#include "interfaces/sycl/SyclWrappedAPI.h"
#define DEVICE_ENABLED
#endif

using namespace device;

DeviceInstance::DeviceInstance() {
  // NOTE: all headers inside of macros define their unique ConcreteInterface.
  // Make sure to not include multiple different interfaces at the same time.
  // Only one interface is allowed per program because of issues of unique compilers, etc.
#ifdef DEVICE_ENABLED
  apiP = std::make_unique<ConcreteAPI>();
  algorithmsP = std::make_unique<Algorithms>();

  algorithmsP->setDeviceApi(apiP);
#endif
}

DeviceInstance::~DeviceInstance() {
#ifdef DEVICE_ENABLED
  this->finalize();
#endif
}

void DeviceInstance::finalize() {
#ifdef DEVICE_ENABLED
  api().finalize();
#endif
}

DeviceInstance& DeviceInstance::instance() {
  static DeviceInstance currentInstance;
  return currentInstance;
}

AbstractAPI& DeviceInstance::api() {
  if (apiP == nullptr) {
    throw std::runtime_error("Device API was called; but it is not initialized.");
  }
  return *apiP;
}

Algorithms& DeviceInstance::algorithms() {
  if (algorithmsP == nullptr) {
    throw std::runtime_error("Device API was called; but it is not initialized.");
  }
  return *algorithmsP;
}
