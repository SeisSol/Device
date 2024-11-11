// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "algorithms/Common.h"
#include "interfaces/sycl/Internals.h"
#include "utils/logger.h"
#include <device.h>

namespace device {
template <typename T> T Algorithms::reduceVector(T *buffer, size_t size, const ReductionType type, void* queuePtr) {
  if (api == nullptr)
    throw std::invalid_argument("api has not been attached to algorithms sub-system");

  auto queue = reinterpret_cast<cl::sycl::queue *>(queuePtr);
  size_t adjustedSize = alignToMultipleOf(size, internals::WARP_SIZE);
  T *reductionBuffer = reinterpret_cast<T*>(api->getStackMemory(adjustedSize * sizeof(T)));
  auto identity = deduceDefaultValue<T>(type);
  this->fillArray(reductionBuffer,
                  identity,
                  adjustedSize,
                  queuePtr);
  api->copyBetween(reductionBuffer, buffer, size * sizeof(T));

  T *redPtr = reinterpret_cast<T*>(api->getStackMemory(sizeof(T)));
  auto rng = internals::computeExecutionRange1D(internals::WARP_SIZE, adjustedSize);

  switch (type) {
  case ReductionType::Add: {
    api->copyTo(redPtr, &identity, sizeof(T));
    auto red = cl::sycl::reduction(redPtr, identity, std::plus<T>());
    queue->submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for(rng, red, [=](cl::sycl::nd_item<1> it, auto &out) {
        out.combine(reductionBuffer[it.get_global_id(0)]);
      });
    });
    break;
  }
  case ReductionType::Max: {
    api->copyTo(redPtr, &identity, sizeof(T));
    auto red = cl::sycl::reduction(redPtr, identity, sycl::ext::oneapi::maximum<T>());
    queue->submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for(rng, red, [=](cl::sycl::nd_item<1> it, auto &out) { 
        out.combine(reductionBuffer[it.get_global_id(0)]);
      });
    });
    break;
  }
  case ReductionType::Min: {
    api->copyTo(redPtr, &identity, sizeof(T));
    auto red = cl::sycl::reduction(redPtr, identity, sycl::ext::oneapi::minimum<T>());
    queue->submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for(rng, red, [=](cl::sycl::nd_item<1> it, auto &out) {
        out.combine(reductionBuffer[it.get_global_id(0)]);
      });
    });
    break;
  }
  default: {
    throw std::invalid_argument("reduction type is not implemented");
  }
  }
  queue->wait();

  T result{};
  api->copyFrom(&result, redPtr, sizeof(T));
  api->popStackMemory();
  api->popStackMemory();

  return result;
}

template unsigned Algorithms::reduceVector(unsigned *buffer, size_t size, ReductionType type, void* streamPtr);
template real Algorithms::reduceVector(real *buffer, size_t size, ReductionType type, void* streamPtr);
} // namespace device

