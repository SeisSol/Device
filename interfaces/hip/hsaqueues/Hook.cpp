// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

// The launch hook for HSA queues (see HsaQueues.h): compile this file into the program itself (not
// into a library), e.g. by linking the CMake target device-hsa-queue-hook. The program then defines
// hipLaunchKernel, which the launches of all kernels reach first (also those from libraries); it
// hands those on HSA streams to the HSA queues, and all others to HIP.

#include "HsaQueues.h"

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

namespace {
device::hsaqueues::LaunchFunction hipLaunch() {
  static const auto launch = []() {
    void* function = dlvsym(RTLD_NEXT, "hipLaunchKernel", "hip_4.2");
    if (function == nullptr) {
      function = dlsym(RTLD_NEXT, "hipLaunchKernel");
    }
    return reinterpret_cast<device::hsaqueues::LaunchFunction>(function);
  }();
  return launch;
}

const bool Registered = []() {
  device::hsaqueues::registerLaunchHook(hipLaunch());
  return true;
}();
} // namespace

extern "C" hipError_t hipLaunchKernel(const void* function,
                                      dim3 grid,
                                      dim3 block,
                                      void** args,
                                      std::size_t sharedMemBytes,
                                      hipStream_t stream) {
  // (once the HSA queues are finalized, HIP reports their handles as invalid)
  if (device::hsaqueues::isTagged(stream) && device::hsaqueues::enabled()) {
    return device::hsaqueues::launch(function, grid, block, args, sharedMemBytes, stream);
  }
  return hipLaunch()(function, grid, block, args, sharedMemBytes, stream);
}
