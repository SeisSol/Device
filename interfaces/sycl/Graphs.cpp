#include "SyclWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"
#include <cassert>
#include <vector>

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
 * Once you have a compute-graph recorded you can invoke it as follows:
 *    launchGraph(graph)                 // 1
 *    syncGraph(graph)                   // 2
 * */

namespace device {
namespace graph_capturing {

} // namespace graph_capturing
} // namespace device


bool ConcreteAPI::isCapableOfGraphCapturing() {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  return true;
#else
  return false;
#endif
}


void ConcreteAPI::streamBeginCapture(std::vector<void*>& streamPtrs) {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  assert(!isCircularStreamsForked && "circular streams must be joined before graph capturing");

  std::vector<sycl::queue> queues;

  for (auto* streamPtr : streamPtrs) {
    queues.emplace_back(*static_cast<sycl::queue*>(streamPtr));
  }

  {
    auto recordingGraph = sycl::ext::oneapi::experimental::command_graph
      <sycl::ext::oneapi::experimental::graph_state::modifiable>(
        queues.at(0).get_context(),
        queues.at(0).get_device()
      );

    graphs.emplace_back(GraphDetails {
      std::nullopt,
      std::move(recordingGraph),
      false
    });
  }

  GraphDetails &graphInstance = graphs.back();

  graphInstance.graph.begin_recording(queues);
#endif
}


void ConcreteAPI::streamEndCapture() {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  auto &graphInstance = graphs.back();
  graphInstance.graph.end_recording();
  graphInstance.instance = std::optional<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>>(graphInstance.graph.finalize());

  graphInstance.ready = true;
#endif
}


DeviceGraphHandle ConcreteAPI::getLastGraphHandle() {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  assert(graphs.back().ready && "a graph has not been fully captured");
  return DeviceGraphHandle(graphs.size() - 1);
#else
  return DeviceGraphHandle();
#endif
}


void ConcreteAPI::launchGraph(DeviceGraphHandle graphHandle, void* streamPtr) {
#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  assert(graphHandle.isInitialized() && "a graph must be captured before launching");
  auto &graphInstance = graphs[graphHandle.getGraphId()];
  static_cast<sycl::queue*>(streamPtr)->submit([&](sycl::handler& handler) {
    handler.ext_oneapi_graph(graphInstance.instance.value());
  });
#endif
}
