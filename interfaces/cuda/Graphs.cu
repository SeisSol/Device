// SPDX-FileCopyrightText: 2023 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "CudaWrappedAPI.h"
#include "DataTypes.h"
#include "Internals.h"
#include "utils/logger.h"

#include <cassert>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <functional>
#include <memory>
#include <vector>

// Explicit graph nodes rest on cudaStreamBeginCaptureToGraph, which the runtime gained in CUDA
// 12.3. Capturing whole streams works without it, so the two get their own macro.
#if defined(DEVICE_USE_GRAPH_CAPTURING) && (CUDART_VERSION >= 12030)
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

namespace device {
struct DeviceGraph {
  cudaGraph_t graph{nullptr};
  cudaGraphExec_t instance{nullptr};

  // one entry per graphAddNode call; an entry may hold zero, one or several native nodes
  std::vector<std::vector<cudaGraphNode_t>> nodes;

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
      cudaGraphExecDestroy(instance);
    }
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
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

  APIWRAP(cudaStreamBeginCapture(static_cast<cudaStream_t>(streamPtrs[0]),
                                 cudaStreamCaptureModeThreadLocal));

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

  APIWRAP(cudaStreamEndCapture(static_cast<cudaStream_t>(graphInstance->streamPtrs[0]),
                               &(graphInstance->graph)));

  APIWRAP(
      cudaGraphInstantiate(&(graphInstance->instance), graphInstance->graph, nullptr, nullptr, 0));

  graphInstance->ready = true;
#endif
}

DeviceGraphHandle ConcreteAPI::graphCreate() {
#ifdef DEVICE_USE_GRAPH_NODES
  auto graphInstance = std::make_shared<DeviceGraph>();
  APIWRAP(cudaGraphCreate(&(graphInstance->graph), 0));
  return DeviceGraphHandle(std::move(graphInstance));
#else
  return DeviceGraphHandle();
#endif
}

namespace {
#ifdef DEVICE_USE_GRAPH_NODES
/**
 * Reads the capture frontier, i.e. the nodes a subsequently captured operation would depend on.
 * Has to be called while the capture is still open.
 *
 * The unversioned name resolves to different signatures depending on the toolkit: up to CUDA
 * 12.x it is the six-argument form, from CUDA 13 on it is the one that also reports edge data.
 * cudaStreamGetCaptureInfo_v2 is not an option, as CUDA 13 no longer declares it.
 */
std::vector<cudaGraphNode_t> captureFrontier(cudaStream_t stream) {
  cudaStreamCaptureStatus captureStatus{};
  unsigned long long captureId{};
  cudaGraph_t capturedGraph{nullptr};
  const cudaGraphNode_t* frontier{nullptr};
  size_t frontierSize{0};

#if CUDART_VERSION >= 13000
  const cudaGraphEdgeData* edgeData{nullptr};
  APIWRAP(cudaStreamGetCaptureInfo(
      stream, &captureStatus, &captureId, &capturedGraph, &frontier, &edgeData, &frontierSize));
#else
  APIWRAP(cudaStreamGetCaptureInfo(
      stream, &captureStatus, &captureId, &capturedGraph, &frontier, &frontierSize));
#endif

  return std::vector<cudaGraphNode_t>(frontier, frontier + frontierSize);
}
#endif
} // namespace

void ConcreteAPI::graphBeginNode(const DeviceGraphHandle& graphHandle,
                                 const std::vector<DeviceGraphNodeHandle>& dependencies,
                                 void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_NODES
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && "a graph must be created before nodes can be added");
  assert(!graphInstance->ready && "no nodes can be added to an instantiated graph");

  std::vector<cudaGraphNode_t> nativeDependencies;
  for (const auto& dependency : dependencies) {
    assert(dependency.isInitialized() && "an uninitialized node cannot be depended upon");
    const auto& nodes = graphInstance->nodes.at(dependency.getNodeId());
    nativeDependencies.insert(nativeDependencies.end(), nodes.begin(), nodes.end());
  }

  APIWRAP(cudaStreamBeginCaptureToGraph(static_cast<cudaStream_t>(streamPtr),
                                        graphInstance->graph,
                                        nativeDependencies.data(),
                                        nullptr,
                                        nativeDependencies.size(),
                                        cudaStreamCaptureModeThreadLocal));
#endif
}

DeviceGraphNodeHandle ConcreteAPI::graphEndNode(const DeviceGraphHandle& graphHandle,
                                                void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_NODES
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && "a node must be opened before it can be closed");

  auto stream = static_cast<cudaStream_t>(streamPtr);
  auto produced = captureFrontier(stream);

  cudaGraph_t endedGraph{nullptr};
  APIWRAP(cudaStreamEndCapture(stream, &endedGraph));
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
      cudaGraphInstantiate(&(graphInstance->instance), graphInstance->graph, nullptr, nullptr, 0));

  graphInstance->ready = true;
#endif
}

void ConcreteAPI::launchGraph(const DeviceGraphHandle& graphHandle, void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  auto* graphInstance = graphHandle.get();
  assert(graphInstance != nullptr && graphInstance->ready &&
         "a graph must be captured before launching");

  APIWRAP(cudaGraphLaunch(graphInstance->instance, static_cast<cudaStream_t>(streamPtr)));
#endif
}
