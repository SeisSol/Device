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


void ConcreteAPI::streamBeginCapture() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  assert(!isCircularStreamsForked && "circular streams must be joined before graph capturing");

  graphs.push_back(GraphDetails{});

  GraphDetails &graphInstance = graphs.back();
  graphInstance.ready = false;

  cudaStreamBeginCapture(defaultStream, cudaStreamCaptureModeGlobal); CHECK_ERR;

  cudaStreamCreateWithFlags(&graphInstance.graphExecutionStream, cudaStreamNonBlocking); CHECK_ERR;
  cudaEventCreate(&(graphInstance.graphCaptureEvent)); CHECK_ERR;

  device::graph_capturing::kernel_firstCapturingKernel<<<1, 1, 0, defaultStream>>>();
  CHECK_ERR;
#endif
}


void ConcreteAPI::streamEndCapture() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  assert(!isCircularStreamsForked && "circular streams must be joined before graph capturing");

  auto& graphInstance = graphs.back();
  cudaStreamEndCapture(defaultStream, &(graphInstance.graph));
  CHECK_ERR;

  cudaGraphInstantiate(&(graphInstance.instance), graphInstance.graph, NULL, NULL, 0);
  CHECK_ERR;

  graphInstance.ready = true;
#endif
}


DeviceGraphHandle ConcreteAPI::getLastGraphHandle() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  assert(graphs.back().ready && "a graph has not been fully captured");
  return DeviceGraphHandle{graphs.size() - 1};
#else
  return DeviceGraphHandle{};
#endif
}


void ConcreteAPI::launchGraph(DeviceGraphHandle graphHandle) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  assert(graphHandle.isInitialized() && "a graph must be captured before launching");
  auto &graphInstance = graphs[graphHandle.graphID];
  cudaGraphLaunch(graphInstance.instance, graphInstance.graphExecutionStream);
  CHECK_ERR;
#endif
}


void ConcreteAPI::syncGraph(DeviceGraphHandle graphHandle) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  assert(graphHandle.isInitialized() && "a graph must be captured before synchronizing");
  auto &graphInstance = graphs[graphHandle.graphID];
  cudaStreamSynchronize(graphInstance.graphExecutionStream);
  CHECK_ERR;
#endif
}
