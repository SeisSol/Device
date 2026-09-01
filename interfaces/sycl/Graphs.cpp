// SPDX-FileCopyrightText: 2023 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DataTypes.h"
#include "Internals.h"
#include "SyclWrappedAPI.h"
#include "utils/logger.h"

#include <cassert>
#include <memory>
#include <vector>

using namespace device;

/* This is a wrapped graph capturing mechanism.
 * Call the following in order to capture a computational graph
 *    auto graph = streamBeginCapture(streams);   // 1
 *    // your GPU code here                       // 2
 *    streamEndCapture(graph);                    // 3
 *
 * Once you have a compute-graph recorded you can invoke it as follows:
 *    launchGraph(graph, stream);                 // 1
 * */

namespace device {
struct DeviceGraph {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  std::optional<sycl::ext::oneapi::experimental::command_graph<
      sycl::ext::oneapi::experimental::graph_state::executable>>
      instance;
  sycl::ext::oneapi::experimental::command_graph<
      sycl::ext::oneapi::experimental::graph_state::modifiable>
      graph;

  DeviceGraph(const sycl::context& context, const sycl::device& device)
      : graph(context, device) {}
#endif

  bool ready{false};

  DeviceGraph(const DeviceGraph&) = delete;
  DeviceGraph& operator=(const DeviceGraph&) = delete;
};
} // namespace device

bool ConcreteAPI::isCapableOfGraphCapturing() {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  return true;
#else
  return false;
#endif
}

DeviceGraphHandle ConcreteAPI::streamBeginCapture(std::vector<void*>& streamPtrs) {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  std::vector<sycl::queue> queues;
  queues.reserve(streamPtrs.size());
  for (auto* streamPtr : streamPtrs) {
    queues.emplace_back(*static_cast<sycl::queue*>(streamPtr));
  }

  auto graphInstance =
      std::make_shared<DeviceGraph>(queues.at(0).get_context(), queues.at(0).get_device());
  graphInstance->graph.begin_recording(queues);

  return DeviceGraphHandle(std::move(graphInstance));
#else
  return DeviceGraphHandle();
#endif
}

void ConcreteAPI::streamEndCapture(const DeviceGraphHandle& handle) {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  auto* graphInstance = handle.get();
  assert(graphInstance != nullptr && "a capture must be started before it can be ended");

  graphInstance->graph.end_recording();
  graphInstance->instance = std::optional<sycl::ext::oneapi::experimental::command_graph<
      sycl::ext::oneapi::experimental::graph_state::executable>>(graphInstance->graph.finalize());

  graphInstance->ready = true;
#endif
}

void ConcreteAPI::launchGraph(const DeviceGraphHandle& graphHandle, void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && graphInstance->ready &&
         "a graph must be captured before launching");

  static_cast<sycl::queue*>(streamPtr)->submit([&](sycl::handler& handler) {
    handler.ext_oneapi_graph(graphInstance->instance.value());
  });
#endif
}
