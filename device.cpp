// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "device.h"

#ifdef DEVICE_LANG_CUDA
#include "interfaces/cuda/CudaWrappedAPI.h"
#elif DEVICE_LANG_HIP
#include "interfaces/hip/HipWrappedAPI.h"
#elif DEVICE_LANG_SYCL
#include "interfaces/sycl/SyclWrappedAPI.h"
#else
#error "Unknown interface for the device wrapper"
#endif

using namespace device;

DeviceInstance::DeviceInstance() {
  // NOTE: all headers inside of macros define their unique ConcreteInterface.
  // Make sure to not include multiple different interfaces at the same time.
  // Only one interface is allowed per program because of issues of unique compilers, etc.
  api = new ConcreteAPI;
  algorithms.setDeviceApi(api);
}

DeviceInstance::~DeviceInstance() {
  this->finalize();
  delete api;
  api = nullptr;
}

void DeviceInstance::finalize() {
  api->finalize();
}

