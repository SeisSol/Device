#include "HipWrappedAPI.h"
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
  isFlagSet<CircularStreamBufferInitialized>(status);
  assert(!isCircularStreamsForked && "circular streams must be joined before graph capturing");

  graphs.push_back(GraphDetails{});

  GraphDetails &graphInstance = graphs.back();
  graphInstance.ready = false;

  hipStreamCreateWithFlags(&graphInstance.graphExecutionStream, hipStreamNonBlocking); CHECK_ERR;
  hipEventCreate(&(graphInstance.graphCaptureEvent)); CHECK_ERR;

  hipStreamBeginCapture(defaultStream, hipStreamCaptureModeGlobal);

  hipLaunchKernelGGL(device::graph_capturing::kernel_firstCapturingKernel, dim3(1), dim3(1), 0, defaultStream);
  CHECK_ERR;
#endif
}


void ConcreteAPI::streamEndCapture() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  isFlagSet<CircularStreamBufferInitialized>(status);
  assert(!isCircularStreamsForked && "circular streams must be joined before graph capturing");

  auto& graphInstance = graphs.back();
  hipStreamEndCapture(defaultStream, &(graphInstance.graph));
  CHECK_ERR;

  hipGraphInstantiate(&(graphInstance.instance), graphInstance.graph, NULL, NULL, 0);
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


void ConcreteAPI::launchGraph(DeviceGraphHandle graphHandle) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  isFlagSet<CircularStreamBufferInitialized>(status);
  assert(graphHandle.isInitialized() && "a graph must be captured before launching");
  auto &graphInstance = graphs[graphHandle.getGraphId()];
  hipGraphLaunch(graphInstance.instance, graphInstance.graphExecutionStream);
  CHECK_ERR;
#endif
}


void ConcreteAPI::syncGraph(DeviceGraphHandle graphHandle) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  isFlagSet<CircularStreamBufferInitialized>(status);
  assert(graphHandle.isInitialized() && "a graph must be captured before synchronizing");
  auto &graphInstance = graphs[graphHandle.getGraphId()];
  hipStreamSynchronize(graphInstance.graphExecutionStream);
  CHECK_ERR;
#endif
}