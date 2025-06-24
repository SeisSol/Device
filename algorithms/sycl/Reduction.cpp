// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "algorithms/Common.h"
#include "interfaces/sycl/Internals.h"
#include "utils/logger.h"
#include <device.h>

#include <sycl/sycl.hpp>

namespace {
  template <typename AccT, typename VecT, typename S> void launchReduction(AccT* result, const VecT *buffer, size_t size, S reducer, void* streamPtr) {
    ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1> { size }, reducer,
        [=](sycl::id<1> idx, auto& redval) {
          redval.combine(static_cast<AccT>(buffer[idx]));
        });
    });
  }
}

namespace device {
template <typename AccT, typename VecT> void Algorithms::reduceVector(AccT* result, const VecT *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr) {
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
      return launchReduction(result, buffer, size, sycl::reduction(result, sycl::plus<AccT>(), properties), streamPtr);
    }
    case ReductionType::Max: {
      return launchReduction(result, buffer, size, sycl::reduction(result, sycl::maximum<AccT>(), properties), streamPtr);
    }
    case ReductionType::Min: {
      return launchReduction(result, buffer, size, sycl::reduction(result, sycl::minimum<AccT>(), properties), streamPtr);
    }
  }
}

template void Algorithms::reduceVector(int* result, const int *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(unsigned* result, const unsigned *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(long* result, const int *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(unsigned long* result, const unsigned *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(long* result, const long *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(unsigned long* result, const unsigned long *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(long long* result, const int *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(unsigned long long* result, const unsigned *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(long long* result, const long *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(unsigned long long* result, const unsigned long *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(long long* result, const long long *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(unsigned long long* result, const unsigned long long *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(float* result, const float *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(double* result, const float *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(double* result, const double *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);

} // namespace device

