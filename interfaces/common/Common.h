// SPDX-FileCopyrightText: 2021 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_COMMON_COMMON_H_
#define SEISSOLDEVICE_INTERFACES_COMMON_COMMON_H_

#include "utils/env.h"
#include "utils/logger.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace device {
enum StatusID {
  DriverApiInitialized = 0,
  DeviceSelected,
  InterfaceInitialized,
  StackMemAllocated,
  Count
};

using StatusT = std::array<bool, StatusID::Count>;

template <StatusID ID>
void isFlagSet(const StatusT& status) {
  assert(status[ID]);
};

template <typename T, typename U>
U align(T number, U alignment) {
  size_t alignmentFactor = (number + alignment - 1) / alignment;
  return alignmentFactor * alignment;
}

/**
 * Maps a stream priority as the device API states it - 0 the lowest, 1 the highest, NaN the
 * runtime default - onto the integer range the runtime reports.
 *
 * CUDA and HIP report that range as a pair in which the numerically *smaller* value is the
 * *higher* priority, so the mapping runs downwards from leastPriority to greatestPriority. Both
 * ends coincide on devices that do not support priorities, which then yields that single value.
 */
inline int mapStreamPriority(int leastPriority, int greatestPriority, double priority) {
  if (std::isnan(priority)) {
    return leastPriority;
  }

  const auto fraction = std::min(std::max(priority, 0.0), 1.0);
  const auto span = static_cast<double>(leastPriority) - static_cast<double>(greatestPriority);
  return static_cast<int>(std::lround(static_cast<double>(leastPriority) - fraction * span));
}
} // namespace device

#endif // SEISSOLDEVICE_INTERFACES_COMMON_COMMON_H_
