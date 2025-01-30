// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "algorithms/Common.h"
#include "interfaces/sycl/Internals.h"
#include "utils/logger.h"
#include <device.h>

namespace device {
template <typename T> struct Sum {
  T defaultValue{deduceDefaultValue<T>(ReductionType::Add)};
  T operator()(T op1, T op2) const { return op1 + op2; }
};

template <typename T> struct Max {
  T defaultValue{deduceDefaultValue<T>(ReductionType::Max)};
  T operator()(T op1, T op2) const { return op1 > op2 ? op1 : op2; }
};

template <typename T> struct Min {
  T defaultValue{deduceDefaultValue<T>(ReductionType::Min)};
  T operator()(T op1, T op2) const { return op1 > op2 ? op2 : op1; }
};

template <typename T, typename OperationT>
size_t reduce(T *vector, size_t reducedSize, OperationT Operation, cl::sycl::queue* queue) {
  auto rng = ::device::internals::computeDefaultExecutionRange1D(reducedSize);

  queue->submit([&](cl::sycl::handler &cgh) {
    auto numLanes = ::device::internals::WARP_SIZE;
    cl::sycl::local_accessor<T> shrMem{numLanes, cgh};

    cgh.parallel_for(rng, [=](cl::sycl::nd_item<> item) {
      auto localRange = item.get_local_range();
      auto gid = item.get_global_id(0);
      auto lid = item.get_local_id(0);

      shrMem[lid] = vector[gid];
      item.barrier();

      for (int offset = numLanes/2; offset >= 1; offset /= 2) {
        if (lid < offset) {
          shrMem[lid] = Operation(shrMem[lid], shrMem[lid + offset]);
        }
        item.barrier();
      }

      auto wid = item.get_group().get_group_id(0);
      if (lid == 0) {
        vector[wid] = shrMem[0];
      }
    });
  });
  return rng.get_local_range().size() / rng.get_local_range().size();
}


template <typename T>
T Algorithms::reduceVector(T *buffer, size_t size, const ReductionType type, void* queuePtr) {
  if (api == nullptr) {
    throw std::invalid_argument("api has not been attached to algorithms sub-system");
  }

  size_t adjustedSize = alignToMultipleOf(size, internals::WARP_SIZE);
  const size_t totalBuffersSize = adjustedSize * sizeof(T);

  T *reductionBuffer = reinterpret_cast<T *>(api->getStackMemory(totalBuffersSize));

  this->fillArray(reductionBuffer,
                  deduceDefaultValue<T>(type),
                  adjustedSize,
                  queuePtr);

  api->copyBetween(reductionBuffer, buffer, size * sizeof(T));

  auto queue = reinterpret_cast<cl::sycl::queue*>(queuePtr);
  size_t numUsedBlocks{0};
  auto launchReduce = [&](size_t size) {
    switch (type) {
    case ReductionType::Add: {
      numUsedBlocks = reduce<T>(reductionBuffer, size, ::device::Sum<T>(), queue);
      break;
    }
    case ReductionType::Max: {
      numUsedBlocks = reduce<T>(reductionBuffer, size, ::device::Max<T>(), queue);
      break;
    }
    case ReductionType::Min: {
      numUsedBlocks = reduce<T>(reductionBuffer, size, ::device::Min<T>(), queue);
      break;
    }
    default: {
      throw std::invalid_argument("reduction type is not implemented");
    }
    }
    return 0;
  };

  for (size_t reducedSize = adjustedSize;
      reducedSize >= internals::WARP_SIZE;
      reducedSize /= internals::WARP_SIZE) {
    launchReduce(reducedSize);
  }

  if (numUsedBlocks > 1) {
      launchReduce(numUsedBlocks);
  }
  queue->wait();

  T result{};
  api->copyFrom(&result, reductionBuffer, sizeof(T));
  this->fillArray(reinterpret_cast<char *>(reductionBuffer),
                  static_cast<char>(0),
                  totalBuffersSize,
                  queuePtr);
  api->popStackMemory();
  return result;
}

template unsigned Algorithms::reduceVector(unsigned *buffer, size_t size, ReductionType type, void* queuePtr);
template real Algorithms::reduceVector(real *buffer, size_t size, ReductionType type, void* queuePtr);
} // namespace device

