// SPDX-FileCopyrightText: 2023-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DataTypes.h"
#include "SyclWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"
#include <cassert>
#include <vector>

using namespace device;

/* This is a wrapped graph capturing CUDA mechanism.
 * Call the following in order to capture a computational graph
 *    streamBeginCapture();              // 1
 *
 *    // your GPU code here              // 2
 *
 *    streamEndCapture();                // 3
 *    auto graph = getGraphInstance();   // 4
 *
 * Once you have a compute-graph recorded you can invoke it as follows:
 *    launchGraph(graph)                 // 1
 *    syncGraph(graph)                   // 2
 * */

namespace device {
namespace graph_capturing {

} // namespace graph_capturing
} // namespace device


bool ConcreteAPI::isCapableOfGraphCapturing() {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  return true;
#else
  return false;
#endif
}


DeviceGraphHandle ConcreteAPI::streamBeginCapture(std::vector<void*>& streamPtrs) {
  auto handle = DeviceGraphHandle();
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  std::vector<sycl::queue> queues;

  for (auto* streamPtr : streamPtrs) {
    queues.emplace_back(*static_cast<sycl::queue*>(streamPtr));
  }

  auto recordingGraph = sycl::ext::oneapi::experimental::command_graph
    <sycl::ext::oneapi::experimental::graph_state::modifiable>(
      queues.at(0).get_context(),
      queues.at(0).get_device()
    );

  {
    std::lock_guard guard(apiMutex);
    graphs.push_back(GraphDetails {
      std::nullopt,
      std::move(recordingGraph),
      false
    });
    handle = DeviceGraphHandle(graphs.size() - 1);

    GraphDetails &graphInstance = graphs[handle.getGraphId()];

    graphInstance.graph.begin_recording(queues);
  }
#endif
  return handle;
}


void ConcreteAPI::streamEndCapture(DeviceGraphHandle handle) {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  std::lock_guard guard(apiMutex);
  auto &graphInstance = graphs[handle.getGraphId()];
  graphInstance.graph.end_recording();
  graphInstance.instance = std::optional<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>>(graphInstance.graph.finalize());

  graphInstance.ready = true;
#endif
}

void ConcreteAPI::launchGraph(DeviceGraphHandle graphHandle, void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  assert(graphHandle.isInitialized() && "a graph must be captured before launching");
  GraphDetails graphInstance = [&]()
  {
    std::lock_guard guard(apiMutex);
    return graphs[graphHandle.getGraphId()];
  }();
  static_cast<sycl::queue*>(streamPtr)->submit([&](sycl::handler& handler) {
    handler.ext_oneapi_graph(graphInstance.instance.value());
  });
#endif
}

