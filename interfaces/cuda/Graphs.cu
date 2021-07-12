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
  assert(!m_isCircularStreamsForked && "circular streams must be joined before graph capturing");

  m_graphs.push_back(GraphDetails{});

  GraphDetails &graphInstance = m_graphs.back();
  graphInstance.ready = false;

  cudaStreamCreateWithFlags(&graphInstance.graphExecutionStream, cudaStreamNonBlocking); CHECK_ERR;
  cudaEventCreate(&(graphInstance.graphCaptureEvent)); CHECK_ERR;

  cudaStreamBeginCapture(m_defaultStream, cudaStreamCaptureModeGlobal);

  device::graph_capturing::kernel_firstCapturingKernel<<<1, 1, 0, m_defaultStream>>>();
  CHECK_ERR;
#endif
}


void ConcreteAPI::streamEndCapture() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  assert(!m_isCircularStreamsForked && "circular streams must be joined before graph capturing");

  auto& graphInstance = m_graphs.back();
  cudaStreamEndCapture(m_defaultStream, &(graphInstance.graph));
  CHECK_ERR;

  cudaGraphInstantiate(&(graphInstance.instance), graphInstance.graph, NULL, NULL, 0);
  CHECK_ERR;

  graphInstance.ready = true;
#endif
}


deviceGraphHandle ConcreteAPI::getLastGraphHandle() {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  assert(m_graphs.back().ready && "a graph has not been fully captured");
  return deviceGraphHandle{m_graphs.size() - 1};
#else
  return deviceGraphHandle{};
#endif
}


void ConcreteAPI::launchGraph(deviceGraphHandle graphHandle) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  assert(graphHandle.isInitialized() && "a graph must be captured before launching");
  auto &graphInstance = m_graphs[graphHandle.graphID];
  cudaGraphLaunch(graphInstance.instance, graphInstance.graphExecutionStream);
  CHECK_ERR;
#endif
}


void ConcreteAPI::syncGraph(deviceGraphHandle graphHandle) {
#ifdef DEVICE_USE_GRAPH_CAPTURING
  assert(graphHandle.isInitialized() && "a graph must be captured before synchronizing");
  auto &graphInstance = m_graphs[graphHandle.graphID];
  cudaStreamSynchronize(graphInstance.graphExecutionStream);
  CHECK_ERR;
#endif
}
