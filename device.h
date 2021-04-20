#ifndef DEVICE_H
#define DEVICE_H

#include "AbstractAPI.h"
#include "Algorithms.h"

namespace device {

// DeviceInstance -> Singleton
class DeviceInstance {
public:
  DeviceInstance(const DeviceInstance &) = delete;
  DeviceInstance &operator=(const DeviceInstance &) = delete;
  static DeviceInstance &getInstance() {
    static DeviceInstance instance;
    return instance;
  }
  ~DeviceInstance() { this->finalize(); }
  void finalize();

  AbstractAPI *api{nullptr};
  Algorithms algorithms{};

private:
  DeviceInstance();
};
} // namespace device
#endif // SEISSOL_DEVICE_H