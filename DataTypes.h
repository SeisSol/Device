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

/**
 * Refers to the set of graph nodes produced by a single AbstractAPI::graphAddNode call.
 *
 * A node handle is an index into the graph that produced it and stays valid for that graph's
 * lifetime. Passing it to a different graph is undefined.
 */
class DeviceGraphNodeHandle {
  public:
  static const size_t invalidId{std::numeric_limits<size_t>::max()};

  DeviceGraphNodeHandle() = default;
  explicit DeviceGraphNodeHandle(size_t id) : nodeId(id) {}

  [[nodiscard]] bool isInitialized() const { return nodeId != invalidId; }

  operator bool() const { return isInitialized(); }

  [[nodiscard]] size_t getNodeId() const { return nodeId; }

  private:
  size_t nodeId{invalidId};
};
} // namespace device

#endif // SEISSOLDEVICE_DATATYPES_H_
