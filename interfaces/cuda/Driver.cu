// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Driver.h"
#include "Internals.h"
#include "utils/logger.h"

#include <cuda_runtime_api.h>
#include <string>

namespace device::internals {

namespace {

/// Looks up the variant of CUDA `version` of the driver function `name`.
template <typename FunctionT>
void lookUp(FunctionT& function, const char* name, unsigned version) {
  void* address = nullptr;
#if CUDART_VERSION >= 12050
  cudaDriverEntryPointQueryResult status{};
  const auto call =
      cudaGetDriverEntryPointByVersion(name, &address, version, cudaEnableDefault, &status);
#elif CUDART_VERSION >= 12000
  // the variant follows from the version of the runtime here, which is not older than `version`
  cudaDriverEntryPointQueryResult status{};
  const auto call = cudaGetDriverEntryPoint(name, &address, cudaEnableDefault, &status);
#else
  const auto call = cudaGetDriverEntryPoint(name, &address, cudaEnableDefault);
#endif
  // where the runtime cannot initialize itself, there is no driver to look in
  const auto result =
      checkResult(call,
                  __FILE__,
                  __LINE__,
                  {cudaErrorInsufficientDriver, cudaErrorNoDevice, cudaErrorInitializationError});
  if (result != cudaSuccess) {
    const auto message = std::string("CUDA cannot be used here: ") + cudaGetErrorString(result) +
                         " (" + std::to_string(result) +
                         "). Is there an NVIDIA GPU with its driver on this node?";
    logError() << message.c_str();
  }
  if (address == nullptr) {
    const auto message = std::string("The CUDA driver has no function ") + name + " of CUDA " +
                         std::to_string(version) + "; it may be older than that.";
    logError() << message.c_str();
  }
  function = reinterpret_cast<FunctionT>(address);
}

DriverApi lookUpAll() {
  DriverApi api;
  lookUp(api.init, "cuInit", 2000);
  lookUp(api.getErrorString, "cuGetErrorString", 6000);
  lookUp(api.deviceGetAttribute, "cuDeviceGetAttribute", 2000);
  lookUp(api.memGetAllocationGranularity, "cuMemGetAllocationGranularity", 10020);
  lookUp(api.memAddressReserve, "cuMemAddressReserve", 10020);
  lookUp(api.memAddressFree, "cuMemAddressFree", 10020);
  lookUp(api.memCreate, "cuMemCreate", 10020);
  lookUp(api.memRelease, "cuMemRelease", 10020);
  lookUp(api.memMap, "cuMemMap", 10020);
  lookUp(api.memUnmap, "cuMemUnmap", 10020);
  lookUp(api.memSetAccess, "cuMemSetAccess", 10020);
  lookUp(api.streamWaitValue32, "cuStreamWaitValue32", 11070);
  return api;
}

} // namespace

const DriverApi& driver() {
  static const DriverApi Api = lookUpAll();
  return Api;
}

} // namespace device::internals
