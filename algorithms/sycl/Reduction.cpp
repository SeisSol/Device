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

template <typename T, typename OperationT> void reduce(size_t size, size_t step, T *buffer, OperationT operation) {

  auto targetSize = getNearestPow2Number(size / (step * 2));
  auto rng = ::device::internals::computeDefaultExecutionRange1D(targetSize);

  getQueue()->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      int j = item.get_global_id(0) * (2 * step);
      if (j < size) {
        T buddyValue = operation.getDefaultValue();
        T currentValue = buffer[j];

        int buddyIndex = j + step;
        if (buddyIndex < size) {
          buddyValue = buffer[buddyIndex];
        }
        buffer[j] = operation(currentValue, buddyValue);
      }
    });
  });
}

template <typename T> T Algorithms::reduceVector(T *buffer, size_t size, const ReductionType type) {
  if (api == nullptr)
    throw std::invalid_argument("api has not been attached to algorithms sub-system");

  for (int step = 1; step < size; step *= 2) {
    switch (type) {
    case ReductionType::Add: {
      reduce<T>(size, step, buffer, ::device::Sum<T>());
      break;
    }
    case ReductionType::Max: {
      reduce<T>(size, step, buffer, ::device::Max<T>());
      break;
    }
    case ReductionType::Min: {
      reduce<T>(size, step, buffer, ::device::Min<T>());
      break;
    }
    default: {
      throw std::invalid_argument("reduction type is not implemented");
    }
    }
  }

  T results{};
  api->copyFrom(&results, buffer, sizeof(T));
  return results;
}

template int Algorithms::reduceVector(int *buffer, size_t size, ReductionType type);
template real Algorithms::reduceVector(real *buffer, size_t size, ReductionType type);
} // namespace device