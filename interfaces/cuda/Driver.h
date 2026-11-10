// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_CUDA_DRIVER_H_
#define SEISSOLDEVICE_INTERFACES_CUDA_DRIVER_H_

#include <cudaTypedefs.h>

namespace device::internals {

/**
 * The functions of the CUDA driver API that the backend calls.
 *
 * The runtime looks them up in the driver (cudaGetDriverEntryPoint), which it loads only when it
 * initializes itself; the library does not link the driver library (libcuda) therefore. A program
 * linked with it starts also where there is no driver, e.g. on the login node of a cluster, as long
 * as it does not use the device there.
 *
 * Each function is the variant of the CUDA version that its type names, which is the variant that
 * the headers of CUDA 11.7 and later map the name of the function to.
 */
struct DriverApi {
  PFN_cuInit_v2000 init{nullptr};
  PFN_cuGetErrorString_v6000 getErrorString{nullptr};
  PFN_cuDeviceGetAttribute_v2000 deviceGetAttribute{nullptr};
  PFN_cuMemGetAllocationGranularity_v10020 memGetAllocationGranularity{nullptr};
  PFN_cuMemAddressReserve_v10020 memAddressReserve{nullptr};
  PFN_cuMemAddressFree_v10020 memAddressFree{nullptr};
  PFN_cuMemCreate_v10020 memCreate{nullptr};
  PFN_cuMemRelease_v10020 memRelease{nullptr};
  PFN_cuMemMap_v10020 memMap{nullptr};
  PFN_cuMemUnmap_v10020 memUnmap{nullptr};
  PFN_cuMemSetAccess_v10020 memSetAccess{nullptr};
  PFN_cuStreamWaitValue32_v11070 streamWaitValue32{nullptr};
};

/**
 * The driver functions. The first call looks them up, which initializes the runtime; it ends the
 * program with an error if there is no driver.
 */
const DriverApi& driver();

} // namespace device::internals

#endif // SEISSOLDEVICE_INTERFACES_CUDA_DRIVER_H_
