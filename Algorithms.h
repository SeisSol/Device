#ifndef DEVICE_ALGORITHMS_HPP
#define DEVICE_ALGORITHMS_HPP

#include "AbstractAPI.h"

namespace device {
enum class ReductionType { Add, Max, Min };
class DeviceInstance;

class Algorithms {
public:
  friend DeviceInstance;
  template <typename T> T reduceVector(T *buffer, size_t size, ReductionType type, void* streamPtr);

  template <typename T> void scaleArray(T *devArray, T scalar, size_t numElements, void* streamPtr);

  template <typename T> void fillArray(T *devArray, T scalar, size_t numElements, void* streamPtr);

  void incrementalAdd(real** out, real *base, size_t increment, size_t numElements, void* streamPtr);

  void touchMemory(real *ptr, size_t size, bool clean, void* streamPtr);

  void touchBatchedMemory(real **basePtr,
                          unsigned elementSize,
                          unsigned numElements,
                          bool clean,
                          void* streamPtr);

  void streamBatchedData(real **baseSrcPtr,
                         real **baseDstPtr,
                         unsigned elementSize,
                         unsigned numElements,
                         void* streamPtr);

  void accumulateBatchedData(real **baseSrcPtr,
                             real **baseDstPtr,
                             unsigned elementSize,
                             unsigned numElements,
                             void* streamPtr);

  template <typename T>
  void copyUniformToScatter(T *src, T **dst, size_t srcOffset, size_t copySize, size_t numElements, void* streamPtr);

  template <typename T>
  void copyScatterToUniform(T **src, T *dst, size_t dstOffset, size_t copySize, size_t numElements, void* streamPtr);

  void compareDataWithHost(const real *hostPtr, const real *devPtr, size_t numElements,
                           const std::string &dataName);

private:
  void setDeviceApi(AbstractAPI *configuredApi) { api = configuredApi; }
  AbstractAPI *api = nullptr;
};
} // namespace device

#endif // DEVICE_ALGORITHMS_HPP
