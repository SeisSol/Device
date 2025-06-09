// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cassert>
#include <cstdint>
#include <device.h>
#include "Algorithms.h"

namespace device {
template <typename T> void Algorithms::scaleArray(T *devArray,
                                                  T scalar,
                                                  const size_t numElements,
                                                  void* streamPtr) {
}
template void Algorithms::scaleArray(float *devArray, float scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(double *devArray, double scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(int *devArray, int scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(unsigned *devArray, unsigned scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(char *devArray, char scalar, const size_t numElements, void* streamPtr);

template <typename T> void Algorithms::fillArray(T *devArray, const T scalar, const size_t numElements, void* streamPtr) {
}
template void Algorithms::fillArray(float *devArray, float scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(double *devArray, double scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(int *devArray, int scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(unsigned *devArray, unsigned scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(char *devArray, char scalar, const size_t numElements, void* streamPtr);

void Algorithms::touchMemoryI(void *ptr, size_t size, bool clean, void* streamPtr) {
}

void Algorithms::incrementalAddI(
  void** out,
  void *base,
  size_t increment,
  size_t numElements,
  void* streamPtr) {
}


  void Algorithms::streamBatchedDataI(const void **baseSrcPtr,
                                    void **baseDstPtr,
                                    size_t elementSize,
                                    size_t numElements,
                                     void* streamPtr) {
  }

  template<typename T>
  void Algorithms::accumulateBatchedData(const T **baseSrcPtr,
                                         T **baseDstPtr,
                                         size_t elementSize,
                                         size_t numElements,
                                         void* streamPtr) {
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

  void Algorithms::touchBatchedMemoryI(void **basePtr,
    size_t elementSize,
                                      size_t numElements,
                                      bool clean,
                                      void* streamPtr) {
  }


template<typename T>
void Algorithms::setToValue(T** out, T value, size_t elementSize, size_t numElements, void* streamPtr) {

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
                                        void* streamPtr) {
  }

  void Algorithms::copyScatterToUniformI(const void **src,
                                        void *dst,
                                        size_t dstOffset,
                                        size_t copySize,
                                        size_t numElements,
                                        void* streamPtr) {
  }


template<typename T>
void Algorithms::compareDataWithHost(const T *hostPtr, const T *devPtr, const size_t numElements,
                                     const std::string &dataName) {

};

template void Algorithms::compareDataWithHost(const float *hostPtr, const float *devPtr, const size_t numElements,
  const std::string &dataName);
template void Algorithms::compareDataWithHost(const double *hostPtr, const double *devPtr, const size_t numElements,
  const std::string &dataName);

template <typename T> void Algorithms::reduceVector(T* result, const T *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr) {
}

template void Algorithms::reduceVector(int* result, const int *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(unsigned* result, const unsigned *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(float* result, const float *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);
template void Algorithms::reduceVector(double* result, const double *buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);

} // namespace device
