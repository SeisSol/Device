// SPDX-FileCopyrightText: 2023-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "CudaWrappedAPI.h"
#include "DataTypes.h"
#include "Internals.h"
#include "utils/logger.h"
#include <cassert>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <mutex>

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
 * Once you have a coompute-graph recorded you can invoke it as follows:
 *    launchGraph(graph)                 // 1
 *    syncGraph(graph)                   // 2
 * */

bool ConcreteAPI::isCapableOfGraphCapturing() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  return true;
#else
  return false;
#endif
}


DeviceGraphHandle ConcreteAPI::streamBeginCapture(std::vector<void*>& streamPtrs) {
  auto handle = DeviceGraphHandle();
#ifdef DEVICE_USE_GRAPH_CAPTURING
  {
    std::lock_guard guard(apiMutex);
    graphs.push_back(GraphDetails{});
    handle = DeviceGraphHandle(graphs.size() - 1);

    GraphDetails &graphInstance = graphs[handle.getGraphId()];  
    graphInstance.ready = false;
    graphInstance.streamPtrs = streamPtrs;
  }

  APIWRAP(cudaStreamBeginCapture(static_cast<cudaStream_t>(streamPtrs[0]), cudaStreamCaptureModeThreadLocal));
  CHECK_ERR;
#endif
  return handle;
}

void ConcreteAPI::streamEndCapture(DeviceGraphHandle handle) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  GraphDetails graphInstance{};
  {
    std::lock_guard guard(apiMutex);
    graphInstance = graphs[handle.getGraphId()];
  }
  APIWRAP(cudaStreamEndCapture(static_cast<cudaStream_t>(graphInstance.streamPtrs[0]), &(graphInstance.graph)));

  APIWRAP(cudaGraphInstantiate(&(graphInstance.instance), graphInstance.graph, nullptr, nullptr, 0));

  graphInstance.ready = true;

  {
    std::lock_guard guard(apiMutex);
    graphs[handle.getGraphId()] = graphInstance;
  }
#endif
}

void ConcreteAPI::launchGraph(DeviceGraphHandle graphHandle, void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  assert(graphHandle.isInitialized() && "a graph must be captured before launching");
  GraphDetails graphInstance{};
  {
    std::lock_guard guard(apiMutex);
    graphInstance = graphs[graphHandle.getGraphId()];
  }
  APIWRAP(cudaGraphLaunch(graphInstance.instance, reinterpret_cast<cudaStream_t>(streamPtr)));
  CHECK_ERR;
#endif
}

