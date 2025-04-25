// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "AbstractAPI.h"
#include "interfaces/sycl/Internals.h"

#include "algorithms/Common.h"

#include <sycl/sycl.hpp>
#include <device.h>

using namespace device::internals;

namespace device {
template <typename T> void Algorithms::scaleArray(T *devArray, T scalar, size_t numElements, void* streamPtr) {
  auto rng = computeExecutionRange1D(device::internals::DefaultBlockDim, numElements);

  ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](sycl::nd_item<> item) {
      size_t index = item.get_global_id(0);
      if (index < numElements) {
        devArray[index] *= scalar;
      }
    });
  });
}
template void Algorithms::scaleArray(float *devArray, float scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(double *devArray, double scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(int *devArray, int scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(unsigned *devArray, unsigned scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(char *devArray, char scalar, const size_t numElements, void* streamPtr);

template <typename T> void Algorithms::fillArray(T *devArray, const T scalar, const size_t numElements, void* streamPtr) {
  auto rng = computeExecutionRange1D(device::internals::DefaultBlockDim, numElements);

  ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](sycl::nd_item<> item) {
      size_t index = item.get_global_id(0);
      if (index < numElements) {
        devArray[index] = scalar;
      }
    });
  });
}

template void Algorithms::fillArray(float *devArray, float scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(double *devArray, double scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(int *devArray, int scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(unsigned *devArray, unsigned scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(char *devArray, char scalar, const size_t numElements, void* streamPtr);

void Algorithms::touchMemoryI(void *ptr, size_t size, bool clean, void* streamPtr) {
  auto rng = computeExecutionRange1D(device::internals::DefaultBlockDim, size);

  ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](sycl::nd_item<> item) {
      int id = item.get_global_id(0);
      if (id < size) {
        if (clean) {
          imemset(ptr, size, id, device::internals::DefaultBlockDim);
        }
        else {
          imemcpy(ptr, ptr, size, id, device::internals::DefaultBlockDim);
        }
      }
    });
  });
}

void Algorithms::incrementalAddI(
  void** out,
  void *base,
  size_t increment,
  size_t numElements,
  void* streamPtr) {

  uintptr_t* oout = reinterpret_cast<uintptr_t*>(out);
  uintptr_t obase = reinterpret_cast<uintptr_t>(base);

  auto rng = computeExecutionRange1D(device::internals::DefaultBlockDim, numElements);

  ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](sycl::nd_item<> item) {
      int id = item.get_global_id(0);
      if (id < numElements) {
        oout[id] = obase + id * increment;
      }
    });
  });
}
} // namespace device

