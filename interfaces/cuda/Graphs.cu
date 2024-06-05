#include "CudaWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"
#include <cassert>

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

namespace device {
namespace graph_capturing {

__global__ void kernel_firstCapturingKernel() {}
} // namespace graph_capturing
} // namespace device


bool ConcreteAPI::isCapableOfGraphCapturing() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  return true;
#else
  return false;
#endif
}


void ConcreteAPI::streamBeginCapture(std::vector<void*>& streamPtrs) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  isFlagSet<CircularStreamBufferInitialized>(status);
  assert(!isCircularStreamsForked && "circular streams must be joined before graph capturing");

  graphs.push_back(GraphDetails{});

  GraphDetails &graphInstance = graphs.back();
  graphInstance.ready = false;

  graphInstance.streamPtr = streamPtrs[0];

  cudaStreamBeginCapture(static_cast<cudaStream_t>(streamPtrs[0]), cudaStreamCaptureModeThreadLocal);
  CHECK_ERR;
#endif
}


void ConcreteAPI::streamEndCapture() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  isFlagSet<CircularStreamBufferInitialized>(status);
  assert(!isCircularStreamsForked && "circular streams must be joined before graph capturing");

  auto& graphInstance = graphs.back();
  cudaStreamEndCapture(static_cast<cudaStream_t>(graphInstance.streamPtr), &(graphInstance.graph));
  CHECK_ERR;

  cudaGraphInstantiate(&(graphInstance.instance), graphInstance.graph, nullptr, nullptr, 0);
  CHECK_ERR;

  graphInstance.ready = true;
#endif
}


DeviceGraphHandle ConcreteAPI::getLastGraphHandle() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  isFlagSet<CircularStreamBufferInitialized>(status);
  assert(graphs.back().ready && "a graph has not been fully captured");
  return DeviceGraphHandle(graphs.size() - 1);
#else
  return DeviceGraphHandle();
#endif
}


void ConcreteAPI::launchGraph(DeviceGraphHandle graphHandle, void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  isFlagSet<CircularStreamBufferInitialized>(status);
  assert(graphHandle.isInitialized() && "a graph must be captured before launching");
  auto &graphInstance = graphs[graphHandle.getGraphId()];
  cudaGraphLaunch(graphInstance.instance, reinterpret_cast<cudaStream_t>(streamPtr));
  CHECK_ERR;
#endif
}
