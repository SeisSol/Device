#include "AbstractAPI.h"
#include "interfaces/sycl/Internals.h"
#include "utils/logger.h"

#include <device.h>
#include <limits>

namespace device {
template <typename T> struct Sum {
  T getDefaultValue() const { return static_cast<T>(0); }
  T operator()(T op1, T op2) const { return op1 + op2; }
};

template <typename T> struct Max {
  T getDefaultValue() const { return std::numeric_limits<T>::min(); }
  T operator()(T op1, T op2) const { return op1 > op2 ? op1 : op2; }
};

template <typename T> struct Min {
  T getDefaultValue() const { return std::numeric_limits<T>::max(); }
  T operator()(T op1, T op2) const { return op1 > op2 ? op2 : op1; }
};

inline size_t getNearestPow2Number(size_t number) {
  auto power = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(number))));
  constexpr size_t BASE = 1;
  return (BASE << power);
}


template <typename T, typename OperationT>
void reduce(T *to, T *from, size_t reducedSize, OperationT Operation, void* queuePtr) {
  auto rng = ::device::internals::computeDefaultExecutionRange1D(reducedSize);
  auto queue = reinterpret_cast<cl::sycl::queue*>(queuePtr);

  queue->submit([&](cl::sycl::handler &cgh) {
    auto numLanes = ::device::internals::WARP_SIZE;
    cl::sycl::local_accessor<T> shrMem{numLanes, cgh};

    cgh.parallel_for(rng, [=](cl::sycl::nd_item<> item) {
      auto localRange = item.get_local_range();
      auto gid = item.get_global_id(0);
      auto lid = item.get_local_id(0);

      shrMem[lid] = from[gid];
      item.barrier();

      for (int offset = numLanes/2; offset >= 1; offset /= 2) {
        if (lid < offset) {
          shrMem[lid] = Operation(shrMem[lid], shrMem[lid + offset]);
        }
        item.barrier();
      }

      auto wid = item.get_group().get_group_id(0);
      if (lid == 0) {
        to[wid] = shrMem[0];
      }
    });
  });
}


template <typename T>
T Algorithms::reduceVector(T *buffer, size_t size, const ReductionType type, void* queuePtr) {
  if (api == nullptr) {
    throw std::invalid_argument("api has not been attached to algorithms sub-system");
  }

  size_t adjustedSize = getNearestPow2Number(size);
  const size_t totalBuffersSize = 2 * adjustedSize * sizeof(T);

  T *reductionBuffer = reinterpret_cast<T *>(api->getStackMemory(totalBuffersSize));

  this->fillArray(reinterpret_cast<char *>(reductionBuffer),
                  static_cast<char>(0),
                  2 * adjustedSize * sizeof(T),
                  queuePtr);

  api->copyBetween(reductionBuffer, buffer, size * sizeof(T));

  T *from = &reductionBuffer[0];
  T *to = &reductionBuffer[adjustedSize];

  size_t swapCounter = 0;
  for (size_t reducedSize = adjustedSize; reducedSize > 0; reducedSize /= internals::WARP_SIZE) {
    switch (type) {
    case ReductionType::Add: {
      reduce<T>(to, from, reducedSize, ::device::Sum<T>(), queuePtr);
      break;
    }
    case ReductionType::Max: {
      reduce<T>(to, from, reducedSize, ::device::Max<T>(), queuePtr);
      break;
    }
    case ReductionType::Min: {
      reduce<T>(to, from, reducedSize, ::device::Min<T>(), queuePtr);
      break;
    }
    default: {
      throw std::invalid_argument("reduction type is not implemented");
    }
    }
    std::swap(to, from);
    ++swapCounter;
  }

  T results{};
  if ((swapCounter % 2) == 0) {
    std::swap(to, from);
  }

  api->copyFrom(&results, to, sizeof(T));
  this->fillArray(reinterpret_cast<char *>(reductionBuffer),
                  static_cast<char>(0),
                  2 * adjustedSize * sizeof(T),
                  queuePtr);
  api->popStackMemory();
  return results;
}

template unsigned Algorithms::reduceVector(unsigned *buffer, size_t size, ReductionType type, void* queuePtr);
template real Algorithms::reduceVector(real *buffer, size_t size, ReductionType type, void* queuePtr);
} // namespace device
