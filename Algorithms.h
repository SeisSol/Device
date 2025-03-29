// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_ALGORITHMS_H_
#define SEISSOLDEVICE_ALGORITHMS_H_

#include "AbstractAPI.h"

namespace device {
enum class ReductionType { Add, Max, Min };
class DeviceInstance;

class Algorithms {
public:
  friend DeviceInstance;
  template <typename T> void reduceVector(T* result, const T* buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr);

  template <typename T> void scaleArray(T *devArray, T scalar, size_t numElements, void* streamPtr);

  template <typename T> void fillArray(T *devArray, T scalar, size_t numElements, void* streamPtr);

  template <typename T>
  void incrementalAdd(T** out, T *base, size_t increment, size_t numElements, void* streamPtr);

  template <typename T>
  void setToValue(T** out, T value, size_t elementSize, size_t numElements, void* streamPtr);

  template <typename T>
  void touchMemory(T *ptr, size_t size, bool clean, void* streamPtr);

  template <typename T>
  void touchBatchedMemory(T **basePtr,
                          unsigned elementSize,
                          unsigned numElements,
                          bool clean,
                          void* streamPtr);

  template <typename T>
  void streamBatchedData(T **baseSrcPtr,
                         T **baseDstPtr,
                         unsigned elementSize,
                         unsigned numElements,
                         void* streamPtr);

  template <typename T>
  void accumulateBatchedData(T **baseSrcPtr,
                             T **baseDstPtr,
                             unsigned elementSize,
                             unsigned numElements,
                             void* streamPtr);

  template <typename T>
  void copyUniformToScatter(T *src, T **dst, size_t srcOffset, size_t copySize, size_t numElements, void* streamPtr);

  template <typename T>
  void copyScatterToUniform(T **src, T *dst, size_t dstOffset, size_t copySize, size_t numElements, void* streamPtr);

  template <typename T>
  void compareDataWithHost(const T *hostPtr, const T *devPtr, size_t numElements,
                           const std::string &dataName);

private:
  void setDeviceApi(AbstractAPI *configuredApi) { api = configuredApi; }
  AbstractAPI *api = nullptr;
};
} // namespace device


#endif // SEISSOLDEVICE_ALGORITHMS_H_

