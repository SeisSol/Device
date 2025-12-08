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

  template<ReductionType Type, typename AtomicRef, typename AccT>
  void atomicUpdate(AtomicRef& atomic, AccT value){

    constexpr auto MO = sycl::memory_order::relaxed;
    AccT expected = value;

    if constexpr(Type == ReductionType::Add) {
      // Explicity pass MO to fetch_add
      atomic.fetch_add(value, MO);
    }
    if constexpr(Type == ReductionType::Max) {
      // sm 60 does not have a fetch max instruction. 
      // Using our own CAS loop
      // Explicity pass MO to load
      // AccT expected = atomic.load(MO);
      
      while(true){
        if(expected>=value) break;

        if(atomic.compare_exchange_weak(expected, value, MO, MO)){
          break;
        }
      }
      // atomic.fetch_max(value);
    }
    if constexpr(Type == ReductionType::Min) {
      //sm 60 does not have a fetch min instruction
      // Using our own CAS loop
      // AccT expected = atomic.load(MO);

      while(true){
        if(expected<=value) break;

        if(atomic.compare_exchange_weak(expected, value, MO, MO)){
          break;
        }
      }
      // atomic.fetch_min(value);
    }
  }

  template <ReductionType Type, typename AccT, typename VecT, typename OpT> void launchReduction(AccT* result, const VecT *buffer, size_t size, OpT operation, bool overrideResult, void* streamPtr) {

    constexpr auto DefaultValue = neutral<Type, AccT>();
    constexpr size_t workGroupSize = 256;
    constexpr size_t itemsPerWorkItem = 8;

    if(overrideResult){
    ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
        cgh.single_task([=](){
          // Initialize the global result to identity value.
          *result = DefaultValue;
        });
      });
    }

    ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {

      size_t numWorkGroups = (size + (workGroupSize * itemsPerWorkItem) - 1)
      / (workGroupSize * itemsPerWorkItem);

      cgh.parallel_for(sycl::nd_range<1> { numWorkGroups*itemsPerWorkItem, workGroupSize },
        [=](sycl::nd_item<1> idx) {

          const auto localId = idx.get_local_id(0);
          const auto groupId = idx.get_group(0);

          //Thread-local reduction
          AccT threadAcc = DefaultValue;
          size_t baseIdx = groupId*(workGroupSize*itemsPerWorkItem) + localId;
          
          #pragma unroll
          for (std::size_t i = 0; i < itemsPerWorkItem*workGroupSize; i += workGroupSize) {
            const auto id = baseIdx + i;
            if(id < size){
              threadAcc = operation(threadAcc, static_cast<AccT>(ntload(&buffer[id])));
            }
          }

          idx.barrier(sycl::access::fence_space::local_space);

          auto reducedValue = sycl::reduce_over_group(idx.get_group(), threadAcc, operation);

          if(localId == 0){
            sycl::atomic_ref<AccT, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device,
                                      sycl::access::address_space::global_space> atomicRes(*result);
            atomicUpdate<Type>(atomicRes, reducedValue);
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

