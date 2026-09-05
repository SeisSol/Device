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
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

// Explicit graph nodes rest on hipStreamBeginCaptureToGraph, which HIP gained in ROCm 6.3.
// Capturing whole streams works without it, so the two get their own macro.
#if defined(DEVICE_USE_GRAPH_CAPTURING) &&                                                         \
    (HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 3))
#define DEVICE_USE_GRAPH_NODES
#endif

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

namespace {
std::mutex hostFunctionMutex;
std::unordered_map<hipGraph_t, std::vector<std::unique_ptr<std::function<void()>>>>
    capturedHostFunctions;
} // namespace

namespace device::internals {
CaptureState captureState(hipStream_t stream) {
  CaptureState state{};
  unsigned long long captureId{};
  const hipGraphNode_t* frontier{nullptr};
  size_t frontierSize{0};

  APIWRAP(hipStreamGetCaptureInfo_v2(
      stream, &state.status, &captureId, &state.graph, &frontier, &frontierSize));

  if (frontier != nullptr) {
    state.frontier.assign(frontier, frontier + frontierSize);
  }
  return state;
}

std::function<void()>* adoptHostFunction(hipGraph_t graph, const std::function<void()>& function) {
  if (graph == nullptr) {
    return nullptr;
  }

  const std::lock_guard<std::mutex> lock(hostFunctionMutex);
  auto& functions = capturedHostFunctions[graph];
  functions.emplace_back(std::make_unique<std::function<void()>>(function));
  return functions.back().get();
}

void forgetHostFunctions(hipGraph_t graph) {
  const std::lock_guard<std::mutex> lock(hostFunctionMutex);
  capturedHostFunctions.erase(graph);
}
} // namespace device::internals

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
    internals::forgetHostFunctions(graph);

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
#ifdef DEVICE_USE_GRAPH_NODES
  return true;
#else
  return false;
#endif
}

DeviceGraphHandle ConcreteAPI::streamBeginCapture(const std::vector<void*>& streamPtrs) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  if (streamPtrs.empty()) {
    logError() << "Graph capturing records streams, so it needs at least one.";
    return DeviceGraphHandle();
  }

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
  assert(graphInstance->instance == nullptr && "a graph is instantiated once");

  APIWRAP(hipStreamEndCapture(static_cast<hipStream_t>(graphInstance->streamPtrs[0]),
                              &(graphInstance->graph)));

  APIWRAP(
      hipGraphInstantiate(&(graphInstance->instance), graphInstance->graph, nullptr, nullptr, 0));

  graphInstance->ready = true;
#endif
}

DeviceGraphHandle ConcreteAPI::graphCreate() {
#ifdef DEVICE_USE_GRAPH_NODES
  auto graphInstance = std::make_shared<DeviceGraph>();
  APIWRAP(hipGraphCreate(&(graphInstance->graph), 0));
  return DeviceGraphHandle(std::move(graphInstance));
#else
  return DeviceGraphHandle();
#endif
}

void ConcreteAPI::graphBeginNode(const DeviceGraphHandle& graphHandle,
                                 const std::vector<DeviceGraphNodeHandle>& dependencies,
                                 void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_NODES
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && "a graph must be created before nodes can be added");
  assert(!graphInstance->ready && "no nodes can be added to an instantiated graph");

  std::vector<hipGraphNode_t> nativeDependencies;
  for (const auto& dependency : dependencies) {
    assert(dependency.isInitialized() && "an uninitialized node cannot be depended upon");
    const auto& nodes = graphInstance->nodes.at(dependency.getNodeId());
    nativeDependencies.insert(nativeDependencies.end(), nodes.begin(), nodes.end());
  }

  // the edge-data argument is not supported by HIP and has to stay a nullptr
  APIWRAP(hipStreamBeginCaptureToGraph(static_cast<hipStream_t>(streamPtr),
                                       graphInstance->graph,
                                       nativeDependencies.data(),
                                       nullptr,
                                       nativeDependencies.size(),
                                       hipStreamCaptureModeThreadLocal));
#endif
}

DeviceGraphNodeHandle ConcreteAPI::graphEndNode(const DeviceGraphHandle& graphHandle,
                                                void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_NODES
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && "a node must be opened before it can be closed");

  auto stream = static_cast<hipStream_t>(streamPtr);
  auto produced = internals::captureState(stream).frontier;

  hipGraph_t endedGraph{nullptr};
  APIWRAP(hipStreamEndCapture(stream, &endedGraph));
  assert(endedGraph == graphInstance->graph && "capturing into a graph hands that same graph back");

  graphInstance->nodes.emplace_back(std::move(produced));
  return DeviceGraphNodeHandle(graphInstance->nodes.size() - 1);
#else
  return DeviceGraphNodeHandle();
#endif
}

void ConcreteAPI::graphInstantiate(const DeviceGraphHandle& graphHandle) {
#ifdef DEVICE_USE_GRAPH_NODES
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && "a graph must be created before it is instantiated");
  assert(graphInstance->instance == nullptr && "a graph is instantiated once");

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
