// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "algorithms/Common.h"
#include "interfaces/sycl/Internals.h"
#include "utils/logger.h"
#include <device.h>

#include <sycl/sycl.hpp>

namespace {
  template <typename T, typename S> void launchReduction(T* result, const T *buffer, size_t size, S reducer, void* streamPtr) {
    ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1> { size }, reducer,
        [=](sycl::id<1> idx, auto& redval) {
          redval.combine(buffer[idx]);
        });
    });
  }
}

namespace device {
template <typename T> void Algorithms::reduceVector(T* result, const T *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr) {
  auto properties = [&]() -> sycl::property_list {
    if (overrideResult) {
      return sycl::property_list{sycl::property::reduction::initialize_to_identity()};
    }
    else {
      return sycl::property_list{};
    }
  }();
  switch (type) {
    case ReductionType::Add: {
      return launchReduction(result, buffer, size, sycl::reduction(result, sycl::plus<>(), properties), streamPtr);
    }
    case ReductionType::Max: {
      return launchReduction(result, buffer, size, sycl::reduction(result, sycl::maximum<>(), properties), streamPtr);
    }
    case ReductionType::Min: {
      return launchReduction(result, buffer, size, sycl::reduction(result, sycl::minimum<>(), properties), streamPtr);
    }
  }
}

template void Algorithms::reduceVector(int* result, const int *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(unsigned* result, const unsigned *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(float* result, const float *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(double* result, const double *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);

} // namespace device

