// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_DATATYPES_H_
#define SEISSOLDEVICE_DATATYPES_H_

#include <cstddef>
#include <limits>

namespace device {
struct DeviceGraphHandle {
  static const size_t invalidId{std::numeric_limits<size_t>::max()};
public:
  explicit DeviceGraphHandle() : graphId(invalidId) {}
  explicit DeviceGraphHandle(size_t id) : graphId(id) {}

  DeviceGraphHandle(const DeviceGraphHandle& other) = default;
  DeviceGraphHandle& operator=(const DeviceGraphHandle& other) = default;

  bool isInitialized() const {
    return graphId != invalidId;
  }

  operator bool() const {
    return isInitialized();
  }

  bool operator!() const {
    return !isInitialized();
  }

  size_t getGraphId() { return graphId; }

private:
  size_t graphId{invalidId};
};
} // namespace device

#endif // SEISSOLDEVICE_DATATYPES_H_

