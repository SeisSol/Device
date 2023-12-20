#include "AbstractAPI.h"
#include "interfaces/sycl/Internals.h"

#include <CL/sycl.hpp>
#include <device.h>

using namespace device::internals;

namespace device {
void Algorithms::streamBatchedData(real **baseSrcPtr, real **baseDstPtr, unsigned elementSize, unsigned numElements, void* streamPtr) {
  auto rng = cl::sycl::nd_range<1>{numElements * device::internals::DefaultBlockDim, device::internals::DefaultBlockDim};

((cl::sycl::queue *) streamPtr)->submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](cl::sycl::nd_item<> item) {
      real *srcElement = baseSrcPtr[item.get_group().get_group_id(0)];
      real *dstElement = baseDstPtr[item.get_group().get_group_id(0)];
#pragma unroll 4
      for (int index = item.get_local_id(0); index < elementSize; index += device::internals::DefaultBlockDim) {
        dstElement[index] = srcElement[index];
      }
    });
  });
}

void Algorithms::accumulateBatchedData(real **baseSrcPtr, real **baseDstPtr, unsigned elementSize,
                                       unsigned numElements, void* streamPtr) {
  auto rng = cl::sycl::nd_range<1>{numElements * device::internals::DefaultBlockDim, device::internals::DefaultBlockDim};

  ((cl::sycl::queue *) streamPtr)->submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](cl::sycl::nd_item<> item) {
      real *srcElement = baseSrcPtr[item.get_group().get_group_id(0)];
      real *dstElement = baseDstPtr[item.get_group().get_group_id(0)];
#pragma unroll 4
      for (int index = item.get_local_id(0); index < elementSize; index += device::internals::DefaultBlockDim) {
        dstElement[index] += srcElement[index];
      }
    });
  });
}

void Algorithms::touchBatchedMemory(real **basePtr, unsigned elementSize, unsigned numElements, bool clean, void* streamPtr) {
  auto rng = cl::sycl::nd_range<1>{numElements * device::internals::DefaultBlockDim, device::internals::DefaultBlockDim};

  ((cl::sycl::queue *) streamPtr)->submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](cl::sycl::nd_item<> item) {
      real *element = basePtr[item.get_group().get_group_id(0)];
      if (element != nullptr) {
#pragma unroll 4
        for (int index = item.get_local_id(0); index < elementSize; index += device::internals::DefaultBlockDim) {
          if (clean) {
            element[index] = 0.0;
          } else {
            real value = element[index];
            // Do something dummy here. We just need to check the pointers point to valid memory locations.
            // Avoid compiler optimization. Possibly, implement a dummy code with asm.
            value += 1.0;
            value -= 1.0;
          }
        }
      }
    });
  });
}

void Algorithms::setToValue(real** out,
                            real value,
                            size_t elementSize,
                            size_t numElements,
                            void* streamPtr) {
  auto rng = cl::sycl::nd_range<1>{numElements * device::internals::DefaultBlockDim, device::internals::DefaultBlockDim};
  ((cl::sycl::queue *) streamPtr)->submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](cl::sycl::nd_item<> item) {
      const auto elementId = item.get_group().get_group_id(0);
      if (elementId < numElements) {
        real *element = out[elementId];
#pragma unroll 4
        for (int i = item.get_local_id(0); i < elementSize; i += device::internals::DefaultBlockDim) {
          element[i] = value;
        }
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
  auto rng = cl::sycl::nd_range<1>{numElements * device::internals::DefaultBlockDim, device::internals::DefaultBlockDim};

  ((cl::sycl::queue *) streamPtr)->submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](cl::sycl::nd_item<> item) {
      T *srcElement = &src[item.get_group().get_group_id(0) * srcOffset];
      T *dstElement = dst[item.get_group().get_group_id(0)];

#pragma unroll 4
      for (int index = item.get_local_id(0); index < copySize; index += device::internals::DefaultBlockDim) {
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
  auto rng = cl::sycl::nd_range<1>{numElements * device::internals::DefaultBlockDim, device::internals::DefaultBlockDim};

  ((cl::sycl::queue *) streamPtr)->submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](cl::sycl::nd_item<> item) {
      T *srcElement = src[item.get_group().get_group_id(0)];
      T *dstElement = &dst[item.get_group().get_group_id(0) * dstOffset];
  
#pragma unroll 4
      for (int index = item.get_local_id(0); index < copySize; index += device::internals::DefaultBlockDim) {
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
