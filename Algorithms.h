// SPDX-FileCopyrightText: 2020 SeisSol Group
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
  template <typename AccT, typename VecT>
  void reduceVector(AccT* result,
                    const VecT* buffer,
                    bool overrideResult,
                    size_t size,
                    ReductionType type,
                    void* streamPtr);

  template <typename T>
  void scaleArray(T* devArray, T scalar, size_t numElements, void* streamPtr);

  template <typename T>
  void fillArray(T* devArray, T scalar, size_t numElements, void* streamPtr);

  void incrementalAddI(
      void** out, void* base, size_t increment, size_t numElements, void* streamPtr);

  template <typename T>
  void incrementalAdd(T** out, T* base, size_t increment, size_t numElements, void* streamPtr) {
    incrementalAddI(reinterpret_cast<void**>(out),
                    reinterpret_cast<void*>(base),
                    increment * sizeof(T),
                    numElements,
                    streamPtr);
  }

  template <typename T>
  void setToValue(T** out, T value, size_t elementSize, size_t numElements, void* streamPtr);

  void touchMemoryI(void* ptr, size_t size, bool clean, void* streamPtr);

  template <typename T>
  void touchMemory(T* ptr, size_t size, bool clean, void* streamPtr) {
    touchMemoryI(reinterpret_cast<void*>(ptr), size * sizeof(T), clean, streamPtr);
  }

  void touchBatchedMemoryI(
      void** basePtr, size_t elementSize, size_t numElements, bool clean, void* streamPtr);

  template <typename T>
  void touchBatchedMemory(
      T** basePtr, size_t elementSize, size_t numElements, bool clean, void* streamPtr) {
    touchBatchedMemoryI(
        reinterpret_cast<void**>(basePtr), elementSize * sizeof(T), numElements, clean, streamPtr);
  }

  void streamBatchedDataI(const void** baseSrcPtr,
                          void** baseDstPtr,
                          size_t elementSize,
                          size_t numElements,
                          void* streamPtr);

  template <typename T>
  void streamBatchedData(const T** baseSrcPtr,
                         T** baseDstPtr,
                         size_t elementSize,
                         size_t numElements,
                         void* streamPtr) {
    streamBatchedDataI(reinterpret_cast<const void**>(baseSrcPtr),
                       reinterpret_cast<void**>(baseDstPtr),
                       elementSize * sizeof(T),
                       numElements,
                       streamPtr);
  }

  template <typename T>
  void accumulateBatchedData(const T** baseSrcPtr,
                             T** baseDstPtr,
                             size_t elementSize,
                             size_t numElements,
                             void* streamPtr);

  void copyUniformToScatterI(const void* src,
                             void** dst,
                             size_t srcOffset,
                             size_t copySize,
                             size_t numElements,
                             void* streamPtr);

  void copyScatterToUniformI(const void** src,
                             void* dst,
                             size_t dstOffset,
                             size_t copySize,
                             size_t numElements,
                             void* streamPtr);

  template <typename T>
  void copyUniformToScatter(const T* src,
                            T** dst,
                            size_t srcOffset,
                            size_t copySize,
                            size_t numElements,
                            void* streamPtr) {
    copyUniformToScatterI(reinterpret_cast<const void*>(src),
                          reinterpret_cast<void**>(dst),
                          srcOffset * sizeof(T),
                          copySize * sizeof(T),
                          numElements,
                          streamPtr);
  }

  template <typename T>
  void copyScatterToUniform(const T** src,
                            T* dst,
                            size_t dstOffset,
                            size_t copySize,
                            size_t numElements,
                            void* streamPtr) {
    copyScatterToUniformI(reinterpret_cast<const void**>(src),
                          reinterpret_cast<void*>(dst),
                          dstOffset * sizeof(T),
                          copySize * sizeof(T),
                          numElements,
                          streamPtr);
  }

  template <typename T>
  void compareDataWithHost(const T* hostPtr,
                           const T* devPtr,
                           size_t numElements,
                           const std::string& dataName);

  private:
  void setDeviceApi(AbstractAPI* configuredApi) { api = configuredApi; }
  AbstractAPI* api = nullptr;
};
} // namespace device

#endif // SEISSOLDEVICE_ALGORITHMS_H_
