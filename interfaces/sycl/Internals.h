// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_SYCL_INTERNALS_H_
#define SEISSOLDEVICE_INTERFACES_SYCL_INTERNALS_H_

#include "device.h"

#include <CL/sycl.hpp>
#include <string>


namespace device {
namespace internals {
constexpr int WARP_SIZE = 32;

constexpr static int DefaultBlockDim = 256;

/*
 * Computes the execution range for a 1D range kernel given a target work group size and a total size.
 * The method determines how many work groups are needed to handle the total size given the local execution size
 * and returns a range to be used within a kernel submission.
 */
inline cl::sycl::nd_range<1> computeExecutionRange1D(const size_t targetWorkGroupSize, const size_t totalSize) {
  size_t factor = (totalSize + targetWorkGroupSize - 1) / targetWorkGroupSize;
  return cl::sycl::nd_range<>{{factor * targetWorkGroupSize}, {targetWorkGroupSize}};
}

/*
 * Computes an execution range using a default size for the work group.
 */
inline cl::sycl::nd_range<1> computeDefaultExecutionRange1D(const size_t totalSize) {
  return computeExecutionRange1D(WARP_SIZE, totalSize);
}

} // namespace internals

} // namespace device

#endif // SEISSOLDEVICE_INTERFACES_SYCL_INTERNALS_H_

