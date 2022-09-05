#include "AbstractAPI.h"
#include "interfaces/sycl/Internals.h"

#include <CL/sycl.hpp>
#include <device.h>

using namespace device::internals;

namespace device {

void Algorithms::streamBatchedData(real **baseSrcPtr, real **baseDstPtr, unsigned elementSize, unsigned numElements, void* streamPtr) {
  auto rng = cl::sycl::nd_range<1>{numElements * 32, 32};

((cl::sycl::queue *) streamPtr)->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      real *srcElement = baseSrcPtr[item.get_group().get_group_id(0)];
      real *dstElement = baseDstPtr[item.get_group().get_group_id(0)];

      for (int index = item.get_local_id(0); index < elementSize; index += item.get_local_range(0)) {
        dstElement[index] = srcElement[index];
      }
    });
  });
}

void Algorithms::accumulateBatchedData(real **baseSrcPtr, real **baseDstPtr, unsigned elementSize,
                                       unsigned numElements, void* streamPtr) {
  auto rng = cl::sycl::nd_range<1>{numElements * 32, 32};

  ((cl::sycl::queue *) streamPtr)->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      real *srcElement = baseSrcPtr[item.get_group().get_group_id(0)];
      real *dstElement = baseDstPtr[item.get_group().get_group_id(0)];
      for (int index = item.get_local_id(0); index < elementSize; index += item.get_local_range(0)) {
        dstElement[index] += srcElement[index];
      }
    });
  });
}

void Algorithms::touchBatchedMemory(real **basePtr, unsigned elementSize, unsigned numElements, bool clean, void* streamPtr) {
  auto rng = cl::sycl::nd_range<1>{numElements * 256, 256};

  ((cl::sycl::queue *) streamPtr)->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      real *element = basePtr[item.get_group().get_group_id(0)];
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
        id += item.get_local_range(0);
      }
    });
  });
}

template <typename T>
void Algorithms::copyUniformToScatter(T *src,
                                      T **dst,
                                      size_t srcOffset,
                                      size_t copySize,
                                      size_t numElements,
                                      void *streamPtr) {
  auto rng = cl::sycl::nd_range<1>{numElements * 256, 256};

  ((cl::sycl::queue *) streamPtr)->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      T *srcElement = &src[item.get_group().get_group_id(0) * srcOffset];
      T *dstElement = dst[item.get_group().get_group_id(0)];
      for (int index = item.get_local_id(0); index < copySize; index += item.get_local_range(0)) {
        dstElement[index] = srcElement[index];
      }
    });
  });
}

template void Algorithms::copyUniformToScatter(real *src,
                                               real **dst,
                                               size_t srcOffset,
                                               size_t copySize,
                                               size_t numElements,
                                               void* streamPtr);

template void Algorithms::copyUniformToScatter(int *src,
                                               int **dst,
                                               size_t srcOffset,
                                               size_t copySize,
                                               size_t numElements,
                                               void* streamPtr);

template void Algorithms::copyUniformToScatter(char *src,
                                               char **dst,
                                               size_t srcOffset,
                                               size_t copySize,
                                               size_t numElements,
                                               void* streamPtr);

template <typename T>
void Algorithms::copyScatterToUniform(T **src,
                                      T *dst,
                                      size_t dstOffset,
                                      size_t copySize,
                                      size_t numElements,
                                      void *streamPtr) {
  auto rng = cl::sycl::nd_range<1>{numElements * 256, 256};

  ((cl::sycl::queue *) streamPtr)->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      T *srcElement = src[item.get_group().get_group_id(0)];
      T *dstElement = &dst[item.get_group().get_group_id(0) * dstOffset];
      for (int index = item.get_local_id(0); index < copySize; index += item.get_local_range(0)) {
        dstElement[index] = srcElement[index];
      }
    });
  });
}

template void Algorithms::copyScatterToUniform(real **src,
                                               real *dst,
                                               size_t dstOffset,
                                               size_t copySize,
                                               size_t numElements,
                                               void* streamPtr);

template void Algorithms::copyScatterToUniform(int **src,
                                               int *dst,
                                               size_t dstOffset,
                                               size_t copySize,
                                               size_t numElements,
                                               void* streamPtr);

template void Algorithms::copyScatterToUniform(char **src,
                                               char *dst,
                                               size_t dstOffset,
                                               size_t copySize,
                                               size_t numElements,
                                               void* streamPtr);

} // namespace device
