// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_COMMON_COMMON_H_
#define SEISSOLDEVICE_INTERFACES_COMMON_COMMON_H_

#include "utils/env.h"
#include <array>
#include <cassert>
#include <cmath>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "utils/logger.h"

namespace device {
enum StatusID {
  DriverApiInitialized = 0,
  DeviceSelected,
  InterfaceInitialized,
  StackMemAllocated,
  Count
};

using StatusT = std::array<bool, StatusID::Count>;

template<StatusID ID>
void isFlagSet(const StatusT& status) { assert(status[ID]); };

template<typename T, typename U>
U align(T number, U alignment) {
  size_t alignmentFactor = (number + alignment - 1) / alignment;
  return alignmentFactor * alignment;
}
} // namespace device

constexpr auto mapPercentage(int minval, int maxval, double value) {
  if (std::isnan(value)) {
    return 0;
  }
  
  const auto convminval = static_cast<double>(minval);
  const auto convmaxval = static_cast<double>(maxval);

  const auto transformed = value * (convmaxval - convminval + 1) + convminval;
  return std::max(std::min(static_cast<int>(std::floor(transformed)), maxval), minval);
}


#endif // SEISSOLDEVICE_INTERFACES_COMMON_COMMON_H_
