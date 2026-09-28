// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "algorithms/Instantiations.h"

#include <algorithm>
#include <cstring>
#include <device.h>

namespace device {
void Algorithms::streamBatchedDataI(const void** baseSrcPtr,
                                    void** baseDstPtr,
                                    size_t elementSize,
                                    size_t numElements,
                                    void* /*streamPtr*/) {
  for (size_t i = 0; i < numElements; ++i) {
    if (baseSrcPtr[i] != nullptr && baseDstPtr[i] != nullptr) {
      std::memmove(baseDstPtr[i], baseSrcPtr[i], elementSize);
    }
  }
}

template <typename T>
void Algorithms::accumulateBatchedData(const T** baseSrcPtr,
                                       T** baseDstPtr,
                                       size_t elementSize,
                                       size_t numElements,
                                       void* /*streamPtr*/) {
  for (size_t i = 0; i < numElements; ++i) {
    const T* srcElement = baseSrcPtr[i];
    T* dstElement = baseDstPtr[i];
    for (size_t j = 0; j < elementSize; ++j) {
      dstElement[j] += srcElement[j];
    }
  }
}
DEVICE_ALGORITHMS_FLOATING_TYPES(DEVICE_ALGORITHMS_INSTANTIATE_ACCUMULATE_BATCHED_DATA)

void Algorithms::touchBatchedMemoryI(
    void** basePtr, size_t elementSize, size_t numElements, bool clean, void* /*streamPtr*/) {
  // without cleaning, there is nothing to do: host memory needs no migration
  if (clean) {
    for (size_t i = 0; i < numElements; ++i) {
      if (basePtr[i] != nullptr) {
        std::memset(basePtr[i], 0, elementSize);
      }
    }
  }
}

template <typename T>
void Algorithms::setToValue(
    T** out, T value, size_t elementSize, size_t numElements, void* /*streamPtr*/) {
  for (size_t i = 0; i < numElements; ++i) {
    std::fill_n(out[i], elementSize, value);
  }
}
DEVICE_ALGORITHMS_VALUE_TYPES(DEVICE_ALGORITHMS_INSTANTIATE_SET_TO_VALUE)

void Algorithms::copyUniformToScatterI(const void* src,
                                       void** dst,
                                       size_t srcOffset,
                                       size_t copySize,
                                       size_t numElements,
                                       void* /*streamPtr*/) {
  const auto* srcBytes = static_cast<const char*>(src);
  for (size_t i = 0; i < numElements; ++i) {
    std::memmove(dst[i], srcBytes + i * srcOffset, copySize);
  }
}

void Algorithms::copyScatterToUniformI(const void** src,
                                       void* dst,
                                       size_t dstOffset,
                                       size_t copySize,
                                       size_t numElements,
                                       void* /*streamPtr*/) {
  auto* dstBytes = static_cast<char*>(dst);
  for (size_t i = 0; i < numElements; ++i) {
    std::memmove(dstBytes + i * dstOffset, src[i], copySize);
  }
}
} // namespace device
