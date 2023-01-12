#include "AbstractAPI.h"
#include "interfaces/sycl/Internals.h"
#include "utils/logger.h"

#include <device.h>
#include <limits>


namespace device {


template <typename T> T Algorithms::reduceVector(T *buffer, size_t size, const ReductionType type, void* queuePtr) {
  if (api == nullptr)
    throw std::invalid_argument("api has not been attached to algorithms sub-system");

  auto queue = reinterpret_cast<cl::sycl::queue *>(queuePtr);
  size_t adjustedSize = internals::WARP_SIZE * ((internals::WARP_SIZE + size - 1) / internals::WARP_SIZE);
  T *reductionBuffer = reinterpret_cast<T *>(api->getStackMemory(adjustedSize));
  this->fillArray(reinterpret_cast<char *>(reductionBuffer),
                  static_cast<char>(0),
                  adjustedSize * sizeof(T),
                  queuePtr);

  api->copyBetween(reductionBuffer, buffer, size * sizeof(T));

  T *redPtr = reinterpret_cast<T *>(api->getStackMemory(sizeof(T)));

  auto rng = internals::computeExecutionRange1D(internals::WARP_SIZE, adjustedSize);
  switch (type) {
  case ReductionType::Add: {
    auto identity = static_cast<T>(0);
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
    auto identity = std::numeric_limits<T>::min();
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
    auto identity = std::numeric_limits<T>::max();
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
  api->syncDevice();

  T results{};
  api->copyFrom(&results, redPtr, sizeof(T));
  api->popStackMemory();
  api->popStackMemory();

  return results;
}

template unsigned Algorithms::reduceVector(unsigned *buffer, size_t size, ReductionType type, void* streamPtr);
template real Algorithms::reduceVector(real *buffer, size_t size, ReductionType type, void* streamPtr);
} // namespace device
