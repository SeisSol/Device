#ifndef DEVICE_INTERNALS_H
#define DEVICE_INTERNALS_H

#include "device.h"

#include <CL/sycl.hpp>
#include <string>

using namespace cl::sycl;

namespace device {
namespace internals {
constexpr int WARP_SIZE = 32;

/*
 * Computes the execution range for a 1D range kernel given a target work group size and a total size.
 * The method determines how many work groups are needed to handle the total size given the local execution size
 * and returns a range to be used within a kernel submission.
 */
inline nd_range<1> computeExecutionRange1D(const size_t TARGET_WORK_GROUP_SIZE, const size_t TOTAL_SIZE) {
  size_t factor = (TOTAL_SIZE + TARGET_WORK_GROUP_SIZE - 1) / TARGET_WORK_GROUP_SIZE;
  return nd_range<>{{factor * TARGET_WORK_GROUP_SIZE}, {TARGET_WORK_GROUP_SIZE}};
}

/*
 * Computes an execution range using a default size for the work group.
 */
inline nd_range<1> computeDefaultExecutionRange1D(const size_t TOTAL_SIZE) {
  return computeExecutionRange1D(WARP_SIZE, TOTAL_SIZE);
}

} // namespace internals

} // namespace device
#endif // DEVICE_INTERNALS_H
