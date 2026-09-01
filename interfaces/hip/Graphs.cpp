// SPDX-FileCopyrightText: 2023 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DataTypes.h"
#include "HipWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"

#include <cassert>
#include <functional>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <vector>

using namespace device;

/* Two ways of building a compute graph are offered.
 *
 * Whole-stream capture, for code that only wants to replay a fixed sequence:
 *    auto graph = streamBeginCapture(streams);   // 1
 *    // your GPU code here                       // 2
 *    streamEndCapture(graph);                    // 3
 *    launchGraph(graph, stream);                 // 4
 *
 * Explicit node construction, for code that knows its own dependency structure:
 *    auto graph = graphCreate();                             // 1
 *    auto a = graphAddNode(graph, {}, stream, recordA);      // 2
 *    auto b = graphAddNode(graph, {a}, stream, recordB);     // 3
 *    graphInstantiate(graph);                                // 4
 *    launchGraph(graph, stream);                             // 5
 * */

namespace device {
struct DeviceGraph {
  hipGraph_t graph{nullptr};
  hipGraphExec_t instance{nullptr};

  // one entry per graphAddNode call; an entry may hold zero, one or several native nodes
  std::vector<std::vector<hipGraphNode_t>> nodes;

  // only used by the whole-stream capture path
  std::vector<void*> streamPtrs;

  bool ready{false};

  DeviceGraph() = default;
  DeviceGraph(const DeviceGraph&) = delete;
  DeviceGraph& operator=(const DeviceGraph&) = delete;

  ~DeviceGraph() {
    // deliberately unchecked: the graph may outlive the device context during teardown, and a
    // failure here has nothing left to report to
    if (instance != nullptr) {
      hipGraphExecDestroy(instance);
    }
    if (graph != nullptr) {
      hipGraphDestroy(graph);
    }
  }
};
} // namespace device

bool ConcreteAPI::isCapableOfGraphCapturing() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  return true;
#else
  return false;
#endif
}

bool ConcreteAPI::isCapableOfGraphNodes() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  // requires hipStreamBeginCaptureToGraph, i.e. ROCm >= 6.3
  return true;
#else
  return false;
#endif
}

DeviceGraphHandle ConcreteAPI::streamBeginCapture(std::vector<void*>& streamPtrs) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  auto graphInstance = std::make_shared<DeviceGraph>();
  graphInstance->streamPtrs = streamPtrs;

  APIWRAP(hipStreamBeginCapture(static_cast<hipStream_t>(streamPtrs[0]),
                                 hipStreamCaptureModeThreadLocal));

  return DeviceGraphHandle(std::move(graphInstance));
#else
  return DeviceGraphHandle();
#endif
}

void ConcreteAPI::streamEndCapture(const DeviceGraphHandle& handle) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  auto* graphInstance = handle.get();
  assert(graphInstance != nullptr && "a capture must be started before it can be ended");

  APIWRAP(hipStreamEndCapture(static_cast<hipStream_t>(graphInstance->streamPtrs[0]),
                               &(graphInstance->graph)));

  APIWRAP(
      hipGraphInstantiate(&(graphInstance->instance), graphInstance->graph, nullptr, nullptr, 0));

  graphInstance->ready = true;
#endif
}

DeviceGraphHandle ConcreteAPI::graphCreate() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  auto graphInstance = std::make_shared<DeviceGraph>();
  APIWRAP(hipGraphCreate(&(graphInstance->graph), 0));
  return DeviceGraphHandle(std::move(graphInstance));
#else
  return DeviceGraphHandle();
#endif
}

DeviceGraphNodeHandle
    ConcreteAPI::graphAddNode(const DeviceGraphHandle& graphHandle,
                              const std::vector<DeviceGraphNodeHandle>& dependencies,
                              void* streamPtr,
                              const std::function<void(void*)>& recorder) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && "a graph must be created before nodes can be added");
  assert(!graphInstance->ready && "no nodes can be added to an instantiated graph");

  std::vector<hipGraphNode_t> nativeDependencies;
  for (const auto& dependency : dependencies) {
    assert(dependency.isInitialized() && "an uninitialized node cannot be depended upon");
    const auto& nodes = graphInstance->nodes.at(dependency.getNodeId());
    nativeDependencies.insert(nativeDependencies.end(), nodes.begin(), nodes.end());
  }

  auto stream = static_cast<hipStream_t>(streamPtr);
  // the edge-data argument is not supported by HIP and has to stay a nullptr
  APIWRAP(hipStreamBeginCaptureToGraph(stream,
                                        graphInstance->graph,
                                        nativeDependencies.data(),
                                        nullptr,
                                        nativeDependencies.size(),
                                        hipStreamCaptureModeThreadLocal));

  recorder(streamPtr);

  // the capture frontier is what the next node has to depend on; it has to be read out before
  // the capture is ended
  hipStreamCaptureStatus captureStatus{};
  unsigned long long captureId{};
  hipGraph_t capturedGraph{nullptr};
  const hipGraphNode_t* frontier{nullptr};
  size_t frontierSize{0};
  APIWRAP(hipStreamGetCaptureInfo_v2(
      stream, &captureStatus, &captureId, &capturedGraph, &frontier, &frontierSize));
  std::vector<hipGraphNode_t> produced(frontier, frontier + frontierSize);

  hipGraph_t endedGraph{nullptr};
  APIWRAP(hipStreamEndCapture(stream, &endedGraph));

  graphInstance->nodes.emplace_back(std::move(produced));
  return DeviceGraphNodeHandle(graphInstance->nodes.size() - 1);
#else
  return DeviceGraphNodeHandle();
#endif
}

void ConcreteAPI::graphInstantiate(const DeviceGraphHandle& graphHandle) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && "a graph must be created before it is instantiated");

  APIWRAP(
      hipGraphInstantiate(&(graphInstance->instance), graphInstance->graph, nullptr, nullptr, 0));

  graphInstance->ready = true;
#endif
}

void ConcreteAPI::launchGraph(const DeviceGraphHandle& graphHandle, void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && graphInstance->ready &&
         "a graph must be captured before launching");

  APIWRAP(hipGraphLaunch(graphInstance->instance, static_cast<hipStream_t>(streamPtr)));
#endif
}
