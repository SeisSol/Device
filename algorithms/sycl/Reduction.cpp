// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "algorithms/Common.h"
#include "interfaces/sycl/Internals.h"
#include "utils/logger.h"
#include <device.h>

#include <limits>
#include <sycl/sycl.hpp>

#if 1

namespace {
  using namespace device;

  template<ReductionType Type, typename T>
  constexpr T neutral() {
    if constexpr(Type == ReductionType::Add) {
      return T(0);
    }
    if constexpr(Type == ReductionType::Max) {
      return std::numeric_limits<T>::min();
    }
    if constexpr(Type == ReductionType::Min) {
      return std::numeric_limits<T>::max();
    }
    return T(0);
  }

  template <ReductionType Type, typename AccT, typename VecT, typename OpT> void launchReduction(AccT* result, const VecT *buffer, size_t size, OpT operation, bool overrideResult, void* streamPtr) {

    constexpr auto DefaultValue = neutral<Type, AccT>();

    ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<AccT, 1> shmem(256, cgh);
      cgh.parallel_for(sycl::nd_range<1> { 1024, 1024 },
        [=](sycl::nd_item<1> idx) {
          const auto subgroup = idx.get_sub_group();
          const auto sgSize = subgroup.get_local_range().size();

          const auto warpCount = subgroup.get_group_range().size();
          const int currentWarp = subgroup.get_group_id();
          const int threadInWarp = subgroup.get_local_id();
          const auto warpsNeeded = (size + sgSize - 1) / sgSize;

          auto acc = DefaultValue;
          
          #pragma unroll 4
          for (std::size_t i = currentWarp; i < warpsNeeded; i += warpCount) {
            const auto id = threadInWarp + i * sgSize;
            auto value = (id < size) ? static_cast<AccT>(ntload(&buffer[id])) : DefaultValue;

            value = sycl::reduce_over_group(subgroup, value, operation);

            acc = operation(acc, value);
          }

          if (threadInWarp == 0) {
            shmem[currentWarp] = acc;
          }

          idx.barrier();

          if (currentWarp == 0) {
            const auto lastWarpsNeeded = (warpCount + sgSize - 1) / sgSize;
            auto lastAcc = DefaultValue;
            #pragma unroll 2
            for (int i = 0; i < lastWarpsNeeded; ++i) {
              const auto id = threadInWarp + i * sgSize;
              auto value = (id < warpCount) ? shmem[id] : DefaultValue;

              value = sycl::reduce_over_group(subgroup, value, operation);

              lastAcc = operation(lastAcc, value);
            }

            if (threadInWarp == 0) {
              if (overrideResult) {
                ntstore(result, lastAcc);
              }
              else {
                ntstore(result, operation(ntload(result), lastAcc));
              }
            }
          }
        });
    });
  }
}

namespace device {
template <typename AccT, typename VecT> void Algorithms::reduceVector(AccT* result, const VecT *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr) {
  switch (type) {
    case ReductionType::Add: {
      return launchReduction<ReductionType::Add>(result, buffer, size, sycl::plus<AccT>(), overrideResult, streamPtr);
    }
    case ReductionType::Max: {
      return launchReduction<ReductionType::Max>(result, buffer, size, sycl::maximum<AccT>(), overrideResult, streamPtr);
    }
    case ReductionType::Min: {
      return launchReduction<ReductionType::Min>(result, buffer, size, sycl::minimum<AccT>(), overrideResult, streamPtr);
    }
  }
}

#else

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

#endif

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

