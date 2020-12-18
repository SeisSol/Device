#ifndef SEISSOL_ALGORITHMS_HPP
#define SEISSOL_ALGORITHMS_HPP

#include "AbstractAPI.h"

namespace device {
enum class ReductionType { Add, Max, Min };
class DeviceInstance;

class Algorithms {
public:
  friend DeviceInstance;
  template <typename T> T reduceVector(T *buffer, size_t size, ReductionType type);

  template <typename T> void scaleArray(T *DevArray, T scalar, size_t numElements);

  template <typename T> void fillArray(T *DevArray, T scalar, size_t numElements);

  void touchMemory(real *ptr, size_t size, bool clean);
  void touchBatchedMemory(real **basePtr, unsigned elementSize, unsigned numElements, bool clean);

  void streamBatchedData(real **baseSrcPtr, real **baseDstPtr, unsigned elementSize,
                         unsigned numElements);
  void accumulateBatchedData(real **baseSrcPtr, real **baseDstPtr, unsigned elementSize,
                             unsigned numElements);

  void compareDataWithHost(const real *hostPtr, const real *devPtr, size_t numElements,
                           const std::string &dataName);

private:
  void setDeviceApi(AbstractAPI *configuredApi) { api = configuredApi; }
  AbstractAPI *api = nullptr;
};
} // namespace device

#endif // SEISSOL_ALGORITHMS_HPP
