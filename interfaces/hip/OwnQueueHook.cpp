// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

// The launch hook for own queues (see OwnQueues.h): compile this file into the program itself (not
// into a library), e.g. by linking the CMake target device-own-queue-hook. The program then defines
// hipLaunchKernel, which the launches of all kernels reach first (also those from libraries); it
// hands those on own streams to the own queues, and all others to HIP.

#include "OwnQueues.h"

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

namespace {
device::ownqueues::LaunchFunction hipLaunch() {
  static const auto launch = []() {
    void* function = dlvsym(RTLD_NEXT, "hipLaunchKernel", "hip_4.2");
    if (function == nullptr) {
      function = dlsym(RTLD_NEXT, "hipLaunchKernel");
    }
    return reinterpret_cast<device::ownqueues::LaunchFunction>(function);
  }();
  return launch;
}

const bool Registered = []() {
  device::ownqueues::registerLaunchHook(hipLaunch());
  return true;
}();
} // namespace

extern "C" hipError_t hipLaunchKernel(const void* function,
                                      dim3 grid,
                                      dim3 block,
                                      void** args,
                                      std::size_t sharedMemBytes,
                                      hipStream_t stream) {
  if (device::ownqueues::isTagged(stream)) {
    return device::ownqueues::launch(function, grid, block, args, sharedMemBytes, stream);
  }
  return hipLaunch()(function, grid, block, args, sharedMemBytes, stream);
}
