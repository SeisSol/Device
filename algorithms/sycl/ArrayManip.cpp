#include "AbstractAPI.h"
#include "interfaces/sycl/Internals.h"

#include <CL/sycl.hpp>
#include <device.h>

using namespace cl::sycl;
using namespace device::internals;

namespace device {

template <typename T> void Algorithms::scaleArray(T *devArray, T scalar, const size_t numElements, void* streamPtr) {
  auto rng = computeExecutionRange1D(64, numElements);

  ((cl::sycl::queue *) streamPtr)->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      size_t index = item.get_global_id(0);
      if (index < numElements) {
        devArray[index] *= scalar;
      }
    });
  });
}

template void Algorithms::scaleArray(real *devArray, real scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(int *devArray, int scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(char *devArray, char scalar, const size_t numElements, void* streamPtr);

template <typename T> void Algorithms::fillArray(T *devArray, const T scalar, const size_t numElements, void* streamPtr) {
  auto rng = computeExecutionRange1D(64, numElements);

  ((cl::sycl::queue *) streamPtr)->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      size_t index = item.get_global_id(0);
      if (index < numElements) {
        devArray[index] = scalar;
      }
    });
  });
}

template void Algorithms::fillArray(real *devArray, real scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(int *devArray, int scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(char *devArray, char scalar, const size_t numElements, void* streamPtr);

void Algorithms::touchMemory(real *ptr, size_t size, bool clean, void* streamPtr) {
  auto rng = computeExecutionRange1D(256, size);

  ((cl::sycl::queue *) streamPtr)->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      int id = item.get_global_id(0);
      if (id < size) {
        if (clean) {
          ptr[id] = 0;
        } else {
          real value = ptr[id];
          // See CUDA for explanation
          value += 1;
          value -= 1;
        }
      }
    });
  });
}
} // namespace device