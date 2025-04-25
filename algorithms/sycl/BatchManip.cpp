// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "AbstractAPI.h"
#include "interfaces/sycl/Internals.h"

#include "algorithms/Common.h"

#include <sycl/sycl.hpp>
#include <device.h>

using namespace device::internals;

namespace device {
void Algorithms::streamBatchedDataI(const void **baseSrcPtr, void **baseDstPtr, size_t elementSize, size_t numElements, void* streamPtr) {
  auto rng = sycl::nd_range<1>{numElements * device::internals::DefaultBlockDim, device::internals::DefaultBlockDim};

((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](sycl::nd_item<> item) {
      const auto block = item.get_group().get_group_id(0);
      const void *srcElement = baseSrcPtr[block];
      void *dstElement = baseDstPtr[block];
      if (srcElement != nullptr && dstElement != nullptr) {
        imemcpy(dstElement, srcElement, elementSize, item.get_local_id(0), device::internals::DefaultBlockDim);
      }
    });
  });
}

template<typename T>
void Algorithms::accumulateBatchedData(const T **baseSrcPtr, T **baseDstPtr, size_t elementSize,
  size_t numElements, void* streamPtr) {
  auto rng = sycl::nd_range<1>{numElements * device::internals::DefaultBlockDim, device::internals::DefaultBlockDim};

  ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](sycl::nd_item<> item) {
      const T *srcElement = baseSrcPtr[item.get_group().get_group_id(0)];
      T *dstElement = baseDstPtr[item.get_group().get_group_id(0)];
#pragma unroll 4
      for (int index = item.get_local_id(0); index < elementSize; index += device::internals::DefaultBlockDim) {
        dstElement[index] += srcElement[index];
      }
    });
  });
}

template void Algorithms::accumulateBatchedData(const float **baseSrcPtr,
  float **baseDstPtr,
  size_t elementSize,
  size_t numElements,
  void* streamPtr);

template void Algorithms::accumulateBatchedData(const double **baseSrcPtr,
  double **baseDstPtr,
  size_t elementSize,
  size_t numElements,
  void* streamPtr);

void Algorithms::touchBatchedMemoryI(void **basePtr, size_t elementSize, size_t numElements, bool clean, void* streamPtr) {
  auto rng = sycl::nd_range<1>{numElements * device::internals::DefaultBlockDim, device::internals::DefaultBlockDim};

  ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](sycl::nd_item<> item) {
      void *element = basePtr[item.get_group().get_group_id(0)];
      if (element != nullptr) {
        if (clean) {
          imemset(element, elementSize, item.get_local_id(0), device::internals::DefaultBlockDim);
        }
        else {
          imemcpy(element, element, elementSize, item.get_local_id(0), device::internals::DefaultBlockDim);
        }
      }
    });
  });
}

template<typename T>
void Algorithms::setToValue(T** out,
                            T value,
                            size_t elementSize,
                            size_t numElements,
                            void* streamPtr) {
  auto rng = sycl::nd_range<1>{numElements * device::internals::DefaultBlockDim, device::internals::DefaultBlockDim};
  ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](sycl::nd_item<> item) {
      const auto elementId = item.get_group().get_group_id(0);
      if (elementId < numElements) {
        T *element = out[elementId];
#pragma unroll 4
        for (int i = item.get_local_id(0); i < elementSize; i += device::internals::DefaultBlockDim) {
          element[i] = value;
        }
      }
    });
  });
}

template void Algorithms::setToValue(float** out, float value, size_t elementSize, size_t numElements, void* streamPtr);
template void Algorithms::setToValue(double** out, double value, size_t elementSize, size_t numElements, void* streamPtr);
template void Algorithms::setToValue(int** out, int value, size_t elementSize, size_t numElements, void* streamPtr);
template void Algorithms::setToValue(unsigned** out, unsigned value, size_t elementSize, size_t numElements, void* streamPtr);
template void Algorithms::setToValue(char** out, char value, size_t elementSize, size_t numElements, void* streamPtr);

void Algorithms::copyUniformToScatterI(const void *src,
                                      void **dst,
                                      size_t srcOffset,
                                      size_t copySize,
                                      size_t numElements,
                                      void *streamPtr) {
  auto rng = sycl::nd_range<1>{numElements * device::internals::DefaultBlockDim, device::internals::DefaultBlockDim};

  ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](sycl::nd_item<> item) {
      const auto block = item.get_group().get_group_id(0);
      const void *srcElement = reinterpret_cast<const void*>(&reinterpret_cast<const char*>(src)[block * srcOffset]);
      void *dstElement = dst[block];
      imemcpy(dstElement, srcElement, copySize, item.get_local_id(0), device::internals::DefaultBlockDim);
    });
  });
}

void Algorithms::copyScatterToUniformI(const void **src,
                                      void *dst,
                                      size_t dstOffset,
                                      size_t copySize,
                                      size_t numElements,
                                      void *streamPtr) {
  auto rng = sycl::nd_range<1>{numElements * device::internals::DefaultBlockDim, device::internals::DefaultBlockDim};

  ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(rng, [=](sycl::nd_item<> item) {
      const auto block = item.get_group().get_group_id(0);
      const void *srcElement = src[block];
      void *dstElement = reinterpret_cast<void*>(&reinterpret_cast<char*>(dst)[block * dstOffset]);
      imemcpy(dstElement, srcElement, copySize, item.get_local_id(0), device::internals::DefaultBlockDim);
    });
  });
}

} // namespace device
