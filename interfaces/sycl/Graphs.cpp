// SPDX-FileCopyrightText: 2023 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DataTypes.h"
#include "Internals.h"
#include "SyclWrappedAPI.h"
#include "utils/logger.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <vector>

using namespace device;

/* Two ways of building a compute graph are offered; see the CUDA backend for the shapes.
 *
 * Both rest on the same mechanism here: the oneAPI graph extension records queues. Whole-queue
 * recording captures everything submitted between begin and end. Node construction records one
 * segment per node and expresses the edges through barriers on the recorded events, which is
 * what the extension offers in place of the node handles the CUDA and HIP backends hand out.
 *
 * The queues record in order, so submissions within one node are chained automatically, and two
 * nodes recorded onto the same queue end up ordered even without an edge between them. Siblings
 * that are meant to run concurrently therefore have to be recorded onto different queues.
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

  // one entry per graphEndNode call; SYCL expresses graph edges through the events of recorded
  // submissions rather than through node objects
  std::vector<std::vector<sycl::event>> nodes;

  // queues that recording has been started on, so that it is only started once per queue
  std::vector<sycl::queue*> recordedQueues;

  DeviceGraph(const sycl::context& context, const sycl::device& device) : graph(context, device) {}
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

bool ConcreteAPI::isCapableOfGraphNodes() {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  return true;
#else
  return false;
#endif
}

DeviceGraphHandle ConcreteAPI::streamBeginCapture(const std::vector<void*>& streamPtrs) {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  if (streamPtrs.empty()) {
    logError() << "Graph capturing records queues, so it needs at least one.";
    return DeviceGraphHandle();
  }

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
  assert(!graphInstance->instance.has_value() && "a graph is instantiated once");

  graphInstance->graph.end_recording();
  graphInstance->instance = std::optional<sycl::ext::oneapi::experimental::command_graph<
      sycl::ext::oneapi::experimental::graph_state::executable>>(graphInstance->graph.finalize());

  graphInstance->ready = true;
#endif
}

DeviceGraphHandle ConcreteAPI::graphCreate() {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  auto& queue = this->currentDefaultQueue();
  return DeviceGraphHandle(std::make_shared<DeviceGraph>(queue.get_context(), queue.get_device()));
#else
  return DeviceGraphHandle();
#endif
}

void ConcreteAPI::graphBeginNode(const DeviceGraphHandle& graphHandle,
                                 const std::vector<DeviceGraphNodeHandle>& dependencies,
                                 void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && "a graph must be created before nodes can be added");
  assert(!graphInstance->ready && "no nodes can be added to an instantiated graph");

  auto* queue = static_cast<sycl::queue*>(streamPtr);
  auto& recorded = graphInstance->recordedQueues;
  if (std::find(recorded.begin(), recorded.end(), queue) == recorded.end()) {
    graphInstance->graph.begin_recording(*queue);
    recorded.push_back(queue);
  }

  std::vector<sycl::event> nativeDependencies;
  for (const auto& dependency : dependencies) {
    assert(dependency.isInitialized() && "an uninitialized node cannot be depended upon");
    const auto& events = graphInstance->nodes.at(dependency.getNodeId());
    nativeDependencies.insert(nativeDependencies.end(), events.begin(), events.end());
  }

  if (!nativeDependencies.empty()) {
    // an empty node that pulls the dependencies onto this queue. Everything the caller records
    // next follows it, because the queues are in order.
    queue->submit([&nativeDependencies](sycl::handler& handler) {
      handler.ext_oneapi_barrier(nativeDependencies);
    });
  }
#endif
}

DeviceGraphNodeHandle ConcreteAPI::graphEndNode(const DeviceGraphHandle& graphHandle,
                                                void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && "a node must be opened before it can be closed");

  auto* queue = static_cast<sycl::queue*>(streamPtr);

  // Closing through a command group rather than through queue::ext_oneapi_submit_barrier: on an
  // in-order queue the shortcut hands back the last recorded event, which is not the barrier's
  // own event if the preceding submission discarded its event.
  std::vector<sycl::event> produced{
      queue->submit([](sycl::handler& handler) { handler.ext_oneapi_barrier(); })};

  graphInstance->nodes.emplace_back(std::move(produced));
  return DeviceGraphNodeHandle(graphInstance->nodes.size() - 1);
#else
  return DeviceGraphNodeHandle();
#endif
}

void ConcreteAPI::graphInstantiate(const DeviceGraphHandle& graphHandle) {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && "a graph must be created before it is instantiated");
  assert(!graphInstance->instance.has_value() && "a graph is instantiated once");

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

  static_cast<sycl::queue*>(streamPtr)->submit(
      [&](sycl::handler& handler) { handler.ext_oneapi_graph(graphInstance->instance.value()); });
#endif
}
