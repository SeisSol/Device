// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_DEVICE_H_
#define SEISSOLDEVICE_DEVICE_H_

#include "AbstractAPI.h"
#include "Algorithms.h"
#include <memory>
#include <stdexcept>

namespace device {

// DeviceInstance -> Singleton
class DeviceInstance {
public:
  DeviceInstance(const DeviceInstance &) = delete;
  DeviceInstance &operator=(const DeviceInstance &) = delete;
  static DeviceInstance& instance();
  ~DeviceInstance();
  void finalize();

  AbstractAPI& api();

  Algorithms& algorithms();

  std::unique_ptr<AbstractAPI> apiP{nullptr};
  std::unique_ptr<Algorithms> algorithmsP{nullptr};

private:
  DeviceInstance();
};
} // namespace device

#endif // SEISSOLDEVICE_DEVICE_H_

