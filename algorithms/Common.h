// SPDX-FileCopyrightText: 2023-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_ALGORITHMS_COMMON_H_
#define SEISSOLDEVICE_ALGORITHMS_COMMON_H_

#include "AbstractAPI.h"
#include "Algorithms.h"
#include <limits>

namespace device {
template <typename T>
T deduceDefaultValue(ReductionType type) {
  if (type == ReductionType::Add) {
    return T{0};
  }
  else if (type == ReductionType::Max) {
    if (std::is_signed<T>::value) {
      return -std::numeric_limits<T>::max();
    }
    else {
      return T{0};
    }
  }
  else if (type == ReductionType::Min) {
    return std::numeric_limits<T>::max();
  }
  return T{0};
}

inline size_t alignToMultipleOf(size_t size, size_t base) {
  return ((size + base - 1) / base) * base;
}
} // namespace


#endif // SEISSOLDEVICE_ALGORITHMS_COMMON_H_

