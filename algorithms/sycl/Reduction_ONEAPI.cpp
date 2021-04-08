#include "AbstractAPI.h"
#include "interfaces/sycl/Internals.h"
#include "utils/logger.h"

#include <device.h>
#include <limits>

namespace device {
inline cl::sycl::queue *getQueue() {
  auto *api = DeviceInstance::getInstance().api;
  return ((cl::sycl::queue *)api->getDefaultStream());
}

template <typename T> T Algorithms::reduceVector(T *buffer, size_t size, const ReductionType type) {
  if (api == nullptr)
    throw std::invalid_argument("api has not been attached to algorithms sub-system");

  auto rng = internals::computeExecutionRange1D(internals::WARP_SIZE, size);
  T *red_ptr = (T *)api->allocGlobMem(sizeof(T));

  switch (type) {
  case ReductionType::Add: {
    auto id = static_cast<T>(0);
    api->copyTo(red_ptr, &id, sizeof(T));
    auto red = cl::sycl::ONEAPI::reduction(red_ptr, red_ptr[0], std::plus<T>());
    getQueue()->submit([&](handler &cgh) {
      cgh.parallel_for(rng, red, [=](nd_item<1> it, auto &out) { out.combine(buffer[it.get_global_id(0)]); });
    });
    break;
  }
  case ReductionType::Max: {
    auto id = std::numeric_limits<T>::min();
    api->copyTo(red_ptr, &id, sizeof(T));
    auto red = cl::sycl::ONEAPI::reduction(red_ptr, red_ptr[0], sycl::ONEAPI::maximum<T>());
    getQueue()->submit([&](handler &cgh) {
      cgh.parallel_for(rng, red, [=](nd_item<1> it, auto &out) { out.combine(buffer[it.get_global_id(0)]); });
    });
    break;
  }
  case ReductionType::Min: {
    auto id = std::numeric_limits<T>::max();
    api->copyTo(red_ptr, &id, sizeof(T));
    auto red = cl::sycl::ONEAPI::reduction(red_ptr, red_ptr[0], sycl::ONEAPI::minimum<T>());
    getQueue()->submit([&](handler &cgh) {
      cgh.parallel_for(rng, red, [=](nd_item<1> it, auto &out) { out.combine(buffer[it.get_global_id(0)]); });
    });
    break;
  }
  default: {
    throw std::invalid_argument("reduction type is not implemented");
  }
  }
  api->synchDevice();

  T results{};
  api->copyFrom(&results, red_ptr, sizeof(T));
  api->freeMem(red_ptr);

  return results;
}

template int Algorithms::reduceVector(int *buffer, size_t size, ReductionType type);
template real Algorithms::reduceVector(real *buffer, size_t size, ReductionType type);
} // namespace device