// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_DEVICE_H_
#define SEISSOLDEVICE_DEVICE_H_

#include "AbstractAPI.h"
#include "Algorithms.h"

#include <memory>

namespace device {

// DeviceInstance -> Singleton
class DeviceInstance {
  public:
  DeviceInstance(const DeviceInstance&) = delete;
  DeviceInstance& operator=(const DeviceInstance&) = delete;

  // defined in device.cpp: all shared objects of a process thus see the same instance, also those
  // which are compiled with hidden visibility (e.g. Python modules)
  static DeviceInstance& instance();

  ~DeviceInstance();
  void finalize();

  // the API and the algorithms exist for the whole lifetime of the instance; the constness of
  // the instance is shallow, i.e. a const instance gives mutable access to them
  [[nodiscard]] AbstractAPI& api() const { return *apiP; }
  [[nodiscard]] Algorithms& algorithms() const { return *algorithmsP; }

  private:
  DeviceInstance();

  std::unique_ptr<AbstractAPI> apiP;
  std::unique_ptr<Algorithms> algorithmsP;
};
} // namespace device

#endif // SEISSOLDEVICE_DEVICE_H_
