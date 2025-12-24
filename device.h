// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_DEVICE_H_
#define SEISSOLDEVICE_DEVICE_H_

#include "AbstractAPI.h"
#include "Algorithms.h"

namespace device {

// DeviceInstance -> Singleton
class DeviceInstance {
  public:
  DeviceInstance(const DeviceInstance&) = delete;
  DeviceInstance& operator=(const DeviceInstance&) = delete;
  static DeviceInstance& getInstance() {
    static DeviceInstance instance;
    return instance;
  }
  ~DeviceInstance();
  void finalize();

  AbstractAPI* api{nullptr};
  Algorithms algorithms{};

  private:
  DeviceInstance();
};
} // namespace device

#endif // SEISSOLDEVICE_DEVICE_H_
