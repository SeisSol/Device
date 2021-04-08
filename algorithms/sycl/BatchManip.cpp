#include "AbstractAPI.h"
#include "interfaces/sycl/Internals.h"

#include <CL/sycl.hpp>
#include <device.h>

using namespace device::internals;

namespace device {

inline cl::sycl::queue *getQueue(void *ptr = nullptr) {
  auto *api = DeviceInstance::getInstance().api;
  if (ptr == nullptr) {
    return ((queue *)api->getDefaultStream());
  }
  return (cl::sycl::queue *)ptr;
}

void Algorithms::streamBatchedData(real **baseSrcPtr, real **baseDstPtr, unsigned elementSize, unsigned numElements) {
  auto rng = computeDefaultExecutionRange1D(numElements);

  getQueue()->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      real *srcElement = baseSrcPtr[item.get_group().get_id(0)];
      real *dstElement = baseDstPtr[item.get_group().get_id(0)];

      for (int index = item.get_local_id(0); index < elementSize; index += item.get_group_range(0)) {
        dstElement[index] = srcElement[index];
      }
    });
  });
}

void Algorithms::accumulateBatchedData(real **baseSrcPtr, real **baseDstPtr, unsigned elementSize,
                                       unsigned numElements) {
  auto rng = computeDefaultExecutionRange1D(numElements);
  getQueue()->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      real *srcElement = baseSrcPtr[item.get_group().get_id(0)];
      real *dstElement = baseDstPtr[item.get_group().get_id(0)];
      for (int index = item.get_local_id(0); index < elementSize; index += item.get_group_range(0)) {
        dstElement[index] += srcElement[index];
      }
    });
  });
}

void Algorithms::touchBatchedMemory(real **basePtr, unsigned elementSize, unsigned numElements, bool clean) {
  auto rng = computeExecutionRange1D(256, numElements);

  getQueue()->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      real *element = basePtr[item.get_group().get_id(0)];
      int id = item.get_local_id(0);
      while (id < elementSize) {
        if (clean) {
          element[id] = 0.0;
        } else {
          real value = element[id];
          // Do something dummy here. We just need to check the pointers point to valid memory locations.
          // Avoid compiler optimization. Possibly, implement a dummy code with asm.
          value += 1.0;
          value -= 1.0;
        }
        id += item.get_group_range(0);
      }
    });
  });
}

template <typename T>
void Algorithms::copyUniformToScatter(T *src, T **dst, size_t chunkSize, size_t numElements, void *streamPtr) {
  auto rng = computeExecutionRange1D(256, numElements);

  getQueue()->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      T *srcElement = &src[item.get_group().get_id(0)];
      T *dstElement = dst[item.get_group().get_id(0)];
      for (int index = item.get_local_id(0); index < chunkSize; index += item.get_group_range(0)) {
        dstElement[index] = srcElement[index];
      }
    });
  });
}

template void Algorithms::copyUniformToScatter(real *src, real **dst, size_t chunkSize, size_t numElements, void* streamPtr);
template void Algorithms::copyUniformToScatter(int *src, int **dst, size_t chunkSize, size_t numElements, void* streamPtr);
template void Algorithms::copyUniformToScatter(char *src, char **dst, size_t chunkSize, size_t numElements, void* streamPtr);

template <typename T>
void Algorithms::copyScatterToUniform(T **src, T *dst, size_t chunkSize, size_t numElements, void *streamPtr) {
  auto rng = computeExecutionRange1D(256, numElements);

  getQueue()->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      T *srcElement = src[item.get_group().get_id(0)];
      T *dstElement = &dst[item.get_group().get_id(0)];
      for (int index = item.get_local_id(0); index < chunkSize; index += item.get_group_range(0)) {
        dstElement[index] = srcElement[index];
      }
    });
  });
}

template void Algorithms::copyScatterToUniform(real **src, real *dst, size_t chunkSize, size_t numElements, void* streamPtr);
template void Algorithms::copyScatterToUniform(int **src, int *dst, size_t chunkSize, size_t numElements, void* streamPtr);
template void Algorithms::copyScatterToUniform(char **src, char *dst, size_t chunkSize, size_t numElements, void* streamPtr);

} // namespace device