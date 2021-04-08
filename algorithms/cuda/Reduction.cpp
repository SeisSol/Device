#include "Reduction.h"
#include "AbstractAPI.h"
#include "DeviceMacros.h"
#include <cassert>
#include <device.h>
#include <limits>
#include <math.h>

namespace device {
inline size_t getNearestPow2Number(size_t number) {
  auto power = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(number))));
  constexpr size_t BASE = 1;
  return (BASE << power);
}

template <typename T> struct Sum {
  __device__ T getDefaultValue() { return static_cast<T>(0); }
  __device__ T operator()(T op1, T op2) { return op1 + op2; }
};

template <typename T> struct Max {
  __device__ T getDefaultValue() { return std::numeric_limits<T>::min(); }
  __device__ T operator()(T op1, T op2) { return op1 > op2 ? op1 : op2; }
};

template <typename T> struct Min {
  __device__ T getDefaultValue() { return std::numeric_limits<T>::max(); }
  __device__ T operator()(T op1, T op2) { return op1 > op2 ? op2 : op1; }
};

template <typename T> T Algorithms::reduceVector(T *buffer, size_t size, const ReductionType type) {
  assert(api != nullptr && "api has not been attached to algorithms sub-system");
  size_t adjustedSize = device::getNearestPow2Number(size);
  const size_t totalBuffersSize = 2 * adjustedSize * sizeof(T);

  T *reductionBuffer = reinterpret_cast<T *>(api->getStackMemory(totalBuffersSize));

  this->fillArray(reinterpret_cast<char *>(reductionBuffer), static_cast<char>(0), 2 * adjustedSize * sizeof(T));
  CHECK_ERR;
  api->copyBetween(reductionBuffer, buffer, size * sizeof(T));

  T *buffer0 = &reductionBuffer[0];
  T *buffer1 = &reductionBuffer[adjustedSize];

  dim3 block(internals::WARP_SIZE, 1, 1);
  dim3 grid = internals::computeGrid1D(internals::WARP_SIZE, size);

  size_t swapCounter = 0;
  for (size_t reducedSize = adjustedSize; reducedSize > 0; reducedSize /= internals::WARP_SIZE) {
    switch (type) {
    case ReductionType::Add: {
      DEVICE_KERNEL_LAUNCH(kernel_reduce, grid, block, 0, 0, buffer1, buffer0, reducedSize, device::Sum<T>());
      break;
    }
    case ReductionType::Max: {
      DEVICE_KERNEL_LAUNCH(kernel_reduce, grid, block, 0, 0, buffer1, buffer0, reducedSize, device::Max<T>());
      break;
    }
    case ReductionType::Min: {
      DEVICE_KERNEL_LAUNCH(kernel_reduce, grid, block, 0, 0, buffer1, buffer0, reducedSize, device::Min<T>());
      break;
    }
    default: {
      assert(false && "reduction type is not implemented");
    }
      CHECK_ERR;
    }
    std::swap(buffer1, buffer0);
    ++swapCounter;
  }

  T results{};
  if ((swapCounter % 2) == 0) {
    api->copyFrom(&results, buffer0, sizeof(T));
  } else {
    api->copyFrom(&results, buffer1, sizeof(T));
  }
  this->fillArray(reinterpret_cast<char *>(reductionBuffer), static_cast<char>(0), 2 * adjustedSize * sizeof(T));
  CHECK_ERR;
  api->popStackMemory();
  return results;
}

template int Algorithms::reduceVector(int *buffer, size_t size, ReductionType type);
template real Algorithms::reduceVector(real *buffer, size_t size, ReductionType type);
} // namespace device