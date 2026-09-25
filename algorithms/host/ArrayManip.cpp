// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "algorithms/Instantiations.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <device.h>

namespace device {
template <typename T>
void Algorithms::scaleArray(T* devArray, T scalar, const size_t numElements, void* /*streamPtr*/) {
  for (size_t i = 0; i < numElements; ++i) {
    devArray[i] = static_cast<T>(devArray[i] * scalar);
  }
}
DEVICE_ALGORITHMS_ARRAY_TYPES(DEVICE_ALGORITHMS_INSTANTIATE_SCALE_ARRAY)

template <typename T>
void Algorithms::fillArray(T* devArray,
                           const T scalar,
                           const size_t numElements,
                           void* /*streamPtr*/) {
  std::fill_n(devArray, numElements, scalar);
}
DEVICE_ALGORITHMS_ARRAY_TYPES(DEVICE_ALGORITHMS_INSTANTIATE_FILL_ARRAY)

void Algorithms::touchMemoryI(void* ptr, size_t size, bool clean, void* /*streamPtr*/) {
  // without cleaning, there is nothing to do: host memory needs no migration
  if (clean) {
    std::memset(ptr, 0, size);
  }
}

void Algorithms::incrementalAddI(
    void** out, void* base, size_t increment, size_t numElements, void* /*streamPtr*/) {
  const auto baseAddress = reinterpret_cast<uintptr_t>(base);
  for (size_t i = 0; i < numElements; ++i) {
    out[i] = reinterpret_cast<void*>(baseAddress + i * increment);
  }
}
} // namespace device
