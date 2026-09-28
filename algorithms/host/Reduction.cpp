// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "algorithms/Instantiations.h"

#include <device.h>
#include <limits>

namespace device {
namespace {
template <typename AccT, typename VecT, typename OperationT>
void reduce(AccT* result,
            const VecT* buffer,
            bool overrideResult,
            size_t size,
            AccT neutral,
            OperationT operation) {
  AccT accumulator = overrideResult ? neutral : *result;
  for (size_t i = 0; i < size; ++i) {
    accumulator = operation(accumulator, static_cast<AccT>(buffer[i]));
  }
  *result = accumulator;
}
} // namespace

template <typename AccT, typename VecT>
void Algorithms::reduceVector(AccT* result,
                              const VecT* buffer,
                              bool overrideResult,
                              size_t size,
                              ReductionType type,
                              void* /*streamPtr*/) {
  switch (type) {
  case ReductionType::Add: {
    reduce(result, buffer, overrideResult, size, AccT{0}, [](AccT a, AccT b) { return a + b; });
    break;
  }
  case ReductionType::Max: {
    reduce(result,
           buffer,
           overrideResult,
           size,
           std::numeric_limits<AccT>::lowest(),
           [](AccT a, AccT b) { return a > b ? a : b; });
    break;
  }
  case ReductionType::Min: {
    reduce(
        result, buffer, overrideResult, size, std::numeric_limits<AccT>::max(), [](AccT a, AccT b) {
          return a > b ? b : a;
        });
    break;
  }
  }
}

DEVICE_ALGORITHMS_REDUCTION_TYPES(DEVICE_ALGORITHMS_INSTANTIATE_REDUCE_VECTOR)

} // namespace device
