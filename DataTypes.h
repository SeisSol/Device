// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_DATATYPES_H_
#define SEISSOLDEVICE_DATATYPES_H_

#include <cstddef>
#include <limits>
#include <memory>

namespace device {

/**
 * Backend-specific payload of a compute graph. Only the active interface implementation defines
 * this type; every other translation unit sees an incomplete type and reaches the graph through
 * DeviceGraphHandle.
 */
struct DeviceGraph;

/**
 * Owning handle to a compute graph.
 *
 * The backend resources (the graph and its executable instance) are released once the last handle
 * pointing to them goes out of scope. A graph that is dropped from a cache therefore also frees
 * its device-side resources.
 */
class DeviceGraphHandle {
  public:
  DeviceGraphHandle() = default;
  explicit DeviceGraphHandle(std::shared_ptr<DeviceGraph> graphPtr) : graph(std::move(graphPtr)) {}

  [[nodiscard]] bool isInitialized() const { return static_cast<bool>(graph); }

  operator bool() const { return isInitialized(); }

  bool operator!() const { return !isInitialized(); }

  [[nodiscard]] DeviceGraph* get() const { return graph.get(); }

  void reset() { graph.reset(); }

  private:
  std::shared_ptr<DeviceGraph> graph;
};
} // namespace device

#endif // SEISSOLDEVICE_DATATYPES_H_
