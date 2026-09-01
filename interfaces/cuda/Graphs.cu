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
  cudaGraph_t graph{nullptr};
  cudaGraphExec_t instance{nullptr};

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

DeviceGraphHandle ConcreteAPI::streamBeginCapture(std::vector<void*>& streamPtrs) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
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

  APIWRAP(cudaStreamEndCapture(static_cast<cudaStream_t>(graphInstance->streamPtrs[0]),
                               &(graphInstance->graph)));

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
