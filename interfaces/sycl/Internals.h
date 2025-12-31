// SPDX-FileCopyrightText: 2021 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_SYCL_INTERNALS_H_
#define SEISSOLDEVICE_INTERFACES_SYCL_INTERNALS_H_

#include "device.h"
#include "utils/logger.h"

#include <string>
#include <sycl/sycl.hpp>

namespace device::internals {
constexpr static int DefaultBlockDim = 256;

template <typename T>
void waitCheck(T&& result) {
  try {
    std::forward<T>(result).wait_and_throw();
  } catch (const sycl::exception& exc) {
    logError() << "SYCL API error:" << std::string(exc.what());
  } catch (const std::exception& exc) {
    logError() << "SYCL API/C++ error:" << std::string(exc.what());
  }
}

/*
 * Computes the execution range for a 1D range kernel given a target work group size and a total
 * size. The method determines how many work groups are needed to handle the total size given the
 * local execution size and returns a range to be used within a kernel submission.
 */
inline sycl::nd_range<1> computeExecutionRange1D(const size_t targetWorkGroupSize,
                                                 const size_t totalSize) {
  size_t factor = (totalSize + targetWorkGroupSize - 1) / targetWorkGroupSize;
  return sycl::nd_range<>{{factor * targetWorkGroupSize}, {targetWorkGroupSize}};
}

} // namespace device::internals

#endif // SEISSOLDEVICE_INTERFACES_SYCL_INTERNALS_H_
