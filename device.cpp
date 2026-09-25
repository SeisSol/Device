// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "device.h"

#ifdef DEVICE_LANG_CUDA
#include "interfaces/cuda/CudaWrappedAPI.h"
#elif DEVICE_LANG_HIP
#include "interfaces/hip/HipWrappedAPI.h"
#elif DEVICE_LANG_SYCL
#include "interfaces/sycl/SyclWrappedAPI.h"
#elif DEVICE_LANG_HOST
#include "interfaces/host/HostWrappedAPI.h"
#else
#error "Unknown interface for the device wrapper"
#endif

#include <memory>

using namespace device;

// NOTE: all headers inside of macros define their unique ConcreteInterface.
// Make sure to not include multiple different interfaces at the same time.
// Only one interface is allowed per program because of issues of unique compilers, etc.
DeviceInstance::DeviceInstance()
    : apiP(std::make_unique<ConcreteAPI>()), algorithmsP(std::make_unique<Algorithms>()) {
  algorithmsP->setDeviceApi(apiP.get());
}

DeviceInstance::~DeviceInstance() { this->finalize(); }

void DeviceInstance::finalize() { apiP->finalize(); }

DeviceInstance& DeviceInstance::instance() {
  static DeviceInstance singleton;
  return singleton;
}
