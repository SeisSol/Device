#include "Reduction.h"
#include "algorithms/Common.h"
#include "DeviceMacros.h"
#include <cassert>
#include <device.h>
#include <math.h>

namespace device {
template <typename T> struct Sum {
  T defaultValue{deduceDefaultValue<T>(ReductionType::Add)};
  __device__ T operator()(T op1, T op2) { return op1 + op2; }
};

template <typename T> struct Max {
  T defaultValue{deduceDefaultValue<T>(ReductionType::Max)};
  __device__ T operator()(T op1, T op2) { return op1 > op2 ? op1 : op2; }
};

template <typename T> struct Min {
  T defaultValue{deduceDefaultValue<T>(ReductionType::Min)};
  __device__ T operator()(T op1, T op2) { return op1 > op2 ? op2 : op1; }
};


template <typename T> T Algorithms::reduceVector(T *buffer, size_t size, const ReductionType type, void* streamPtr) {
  assert(api != nullptr && "api has not been attached to algorithms sub-system");
  size_t adjustedSize = device::alignToMultipleOf(size, internals::WARP_SIZE);
  const size_t totalBuffersSize = adjustedSize * sizeof(T);
  T *reductionBuffer = reinterpret_cast<T *>(api->getStackMemory(totalBuffersSize));

  this->fillArray(reductionBuffer,
                  deduceDefaultValue<T>(type),
                  adjustedSize,
                  streamPtr);
  CHECK_ERR;

  api->copyBetween(reductionBuffer, buffer, size * sizeof(T));

  dim3 block(internals::WARP_SIZE, 1, 1);
  auto stream = reinterpret_cast<internals::deviceStreamT>(streamPtr);

  auto launchReduce = [&](dim3& grid, size_t size) {
    switch (type) {
    case ReductionType::Add: {
      DEVICE_KERNEL_LAUNCH(kernel_reduce, grid, block, 0, stream, reductionBuffer, size, device::Sum<T>());
      break;
    }
    case ReductionType::Max: {
      DEVICE_KERNEL_LAUNCH(kernel_reduce, grid, block, 0, stream, reductionBuffer, size, device::Max<T>());
      break;
    }
    case ReductionType::Min: {
      DEVICE_KERNEL_LAUNCH(kernel_reduce, grid, block, 0, stream, reductionBuffer, size, device::Min<T>());
      break;
    }
    default: {
      assert(false && "reduction type is not implemented");
    }
      CHECK_ERR;
    }
  };

  dim3 grid{};
  for (size_t reducedSize = adjustedSize;
      reducedSize >= internals::WARP_SIZE;
      reducedSize /= internals::WARP_SIZE) {
    grid = internals::computeGrid1D(internals::WARP_SIZE, reducedSize);
    launchReduce(grid, reducedSize);
  }

  if (grid.x > 1) {
      auto reducedSize = static_cast<size_t>(grid.x);
      grid = dim3(1, 1, 1);
      launchReduce(grid, reducedSize);
  }

  T result{};
  api->syncStreamWithHost(streamPtr);
  api->copyFrom(&result, reductionBuffer, sizeof(T));
  this->fillArray(reinterpret_cast<char *>(reductionBuffer),
                  static_cast<char>(0),
                  totalBuffersSize,
                  streamPtr);
  CHECK_ERR;
  api->popStackMemory();
  return result;
}

template unsigned Algorithms::reduceVector(unsigned *buffer, size_t size, ReductionType type, void* streamPtr);
template real Algorithms::reduceVector(real *buffer, size_t size, ReductionType type, void* streamPtr);
} // namespace device
