// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_ABSTRACTAPI_H_
#define SEISSOLDEVICE_ABSTRACTAPI_H_

#include "DataTypes.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace device {

enum class Destination { Host, CurrentDevice };
enum class ProfilingColors : uint32_t {
  Black = 0x000000,
  White = 0xFFFFFF,
  Red = 0xFF0000,
  Lime = 0x00FF00,
  Blue = 0x0000FF,
  Yellow = 0xFFFF00,
  Cyan = 0x00FFFF,
  Magenta = 0xFF00FF,
  Count = 8
};

struct AbstractAPI {
  virtual ~AbstractAPI() = default;

  /**
   * Selects the device for the calling thread and makes it the choice of the process. Backends
   * that keep the selected device per thread give a thread that has not called this the device
   * the process selected, the first time that thread asks for the device id.
   */
  virtual void setDevice(int deviceId) = 0;
  virtual int getDeviceId() = 0;

  virtual int getNumDevices() = 0;
  virtual unsigned getGlobMemAlignment() = 0;
  virtual std::string getDeviceInfoAsText(int deviceId) = 0;
  virtual void syncDevice() = 0;

  virtual std::string getApiName() = 0;
  virtual std::string getDeviceName(int deviceId) = 0;
  virtual std::string getPciAddress(int deviceId) = 0;

  virtual void* allocGlobMem(size_t size, bool compress = false) = 0;
  virtual void* allocUnifiedMem(size_t size,
                                bool compress = false,
                                Destination hint = Destination::CurrentDevice) = 0;
  virtual void*
      allocPinnedMem(size_t size, bool compress = false, Destination hint = Destination::Host) = 0;
  virtual void freeGlobMem(void* devPtr) = 0;
  virtual void freeUnifiedMem(void* devPtr) = 0;
  virtual void freePinnedMem(void* devPtr) = 0;
  virtual std::string getMemLeaksReport() = 0;

  virtual void* allocMemAsync(size_t size, void* streamPtr) = 0;
  virtual void freeMemAsync(void* devPtr, void* streamPtr) = 0;

  virtual void pinMemory(void* ptr, size_t size) = 0;
  virtual void unpinMemory(void* ptr) = 0;
  virtual void* devicePointer(void* ptr) = 0;

  virtual void copyTo(void* dst, const void* src, size_t count) = 0;
  virtual void copyFrom(void* dst, const void* src, size_t count) = 0;
  virtual void copyBetween(void* dst, const void* src, size_t count) = 0;
  virtual void copyToAsync(void* dst, const void* src, size_t count, void* streamPtr) = 0;
  virtual void copyFromAsync(void* dst, const void* src, size_t count, void* streamPtr) = 0;
  virtual void copyBetweenAsync(void* dst, const void* src, size_t count, void* streamPtr) = 0;

  virtual void copy2dArrayTo(
      void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) = 0;
  virtual void copy2dArrayFrom(
      void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) = 0;
  virtual void
      prefetchUnifiedMemTo(Destination type, const void* devPtr, size_t count, void* streamPtr) = 0;

  virtual size_t getMaxAvailableMem() = 0;
  virtual size_t getCurrentlyOccupiedMem() = 0;
  virtual size_t getCurrentlyOccupiedUnifiedMem() = 0;

  virtual void* getDefaultStream() = 0;
  virtual void syncDefaultStreamWithHost() = 0;

  virtual bool isCapableOfGraphCapturing() = 0;
  virtual DeviceGraphHandle streamBeginCapture(const std::vector<void*>& streamPtrs) = 0;
  virtual void streamEndCapture(const DeviceGraphHandle& handle) = 0;
  virtual void launchGraph(const DeviceGraphHandle& graphHandle, void* streamPtr) = 0;

  /**
   * Explicit graph construction.
   *
   * Instead of recording a whole stream and letting the backend infer the dependency structure
   * from events, the caller states the structure directly: every graphAddNode call contributes
   * the work recorded by `recorder` and makes it depend on exactly `dependencies`. Fork/join is
   * then a property of the graph rather than something that has to be expressed through streams
   * and events.
   *
   * A single graph is built by one thread at a time. `recorder` receives a stream that is only a
   * recording vehicle: what it enqueues becomes the node, and the dependencies stated in the call
   * are the ones the node is guaranteed to get.
   *
   * Nodes that are meant to run concurrently have to be recorded onto different streams. A stream
   * can be reused for a later node, but a backend that expresses edges through the recorded
   * stream rather than through node handles - the SYCL one does, since the graph extension has no
   * node handles to hand out - adds an edge between two nodes that shared a stream, and those two
   * then run one after the other.
   *
   * If `recorder` enqueues nothing, the returned handle refers to `dependencies` themselves, so
   * an empty recorder is a valid way to express a pure join node.
   *
   * graphBeginNode and graphEndNode are the same thing split in two, for callers that cannot
   * wrap the recorded work in a callback and instead have to leave a node open across code they
   * do not control. Only one node per stream may be open at a time.
   */
  virtual bool isCapableOfGraphNodes() = 0;
  virtual DeviceGraphHandle graphCreate() = 0;
  virtual void graphBeginNode(const DeviceGraphHandle& graphHandle,
                              const std::vector<DeviceGraphNodeHandle>& dependencies,
                              void* streamPtr) = 0;
  virtual DeviceGraphNodeHandle graphEndNode(const DeviceGraphHandle& graphHandle,
                                             void* streamPtr) = 0;
  virtual void graphInstantiate(const DeviceGraphHandle& graphHandle) = 0;

  /**
   * As graphBeginNode, for work that has to be captured into a graph of its own, which then becomes
   * a single node of the graph; graphEndNode ends it as usual. For libraries that keep state per
   * capture, and thus fail when they get captured into the same graph several times: CUDA gives
   * all captures into one graph the same id, which NCCL takes for one capture all along.
   *
   * Backends that tell such captures apart (HIP) record an ordinary node.
   */
  virtual void graphBeginIsolatedNode(const DeviceGraphHandle& graphHandle,
                                      const std::vector<DeviceGraphNodeHandle>& dependencies,
                                      void* streamPtr) {
    graphBeginNode(graphHandle, dependencies, streamPtr);
  }

  DeviceGraphNodeHandle graphAddNode(const DeviceGraphHandle& graphHandle,
                                     const std::vector<DeviceGraphNodeHandle>& dependencies,
                                     void* streamPtr,
                                     const std::function<void(void*)>& recorder) {
    graphBeginNode(graphHandle, dependencies, streamPtr);
    recorder(streamPtr);
    return graphEndNode(graphHandle, streamPtr);
  }

  /**
   * Creates a stream. `priority` runs from 0 for the lowest to 1 for the highest priority the
   * device offers; NAN asks for the runtime default. Backends that only know a few priority
   * classes round to the nearest one, and a device without priority support ignores the value.
   */
  virtual void* createStream(double priority = NAN) = 0;

  /**
   * Streams on queues of their own: the device runs the work of such a stream on a hardware queue
   * that only the streams with the same queue number share, in the order it gets enqueued, so that
   * a stream that waits holds up no other ones. (HIP puts all streams onto a few hardware queues
   * that they share; its backend runs own streams on HSA queues outside of HIP, see
   * interfaces/hip/OwnQueues.h, and needs the target device-own-queue-hook in the program.)
   *
   * Enables them, and returns whether the device has them; call it before any stream or event gets
   * created. Then, events can be recorded on and waited for by streams of either kind; events time
   * only what is between two of them recorded on own streams, and only on request (HIP:
   * DEVICE_OWN_QUEUE_TIMING=1, see isCapableOfEventTiming()).
   */
  virtual bool enableOwnQueues() { return false; }

  /**
   * A stream on the own queue `queue` (see enableOwnQueues()); without own queues, an ordinary one.
   * A priority (0 lowest, 1 highest) applies to the whole queue: its work gets taken up before that
   * of queues of a lower priority.
   */
  virtual void* createStreamOnOwnQueue(int /*queue*/, double /*priority*/ = NAN) {
    return createStream();
  }

  virtual void destroyGenericStream(void* streamPtr) = 0;
  virtual void syncStreamWithHost(void* streamPtr) = 0;
  virtual bool isStreamWorkDone(void* streamPtr) = 0;
  virtual void syncStreamWithEvent(void* streamPtr, void* eventPtr) = 0;
  virtual void streamHostFunction(void* streamPtr, const std::function<void()>& function) = 0;

  /**
   * Blocks the stream until the value at `location` has reached at least `value`. `location` has
   * to be host memory that the device can read, i.e. an allocation from allocPinnedMem with
   * Destination::CurrentDevice, or a counter from allocCounter().
   */
  virtual void streamWaitMemory(void* streamPtr, uint32_t* location, uint32_t value) = 0;

  /**
   * A counter in host memory, 0 at first, that the host and the device (by its address from
   * devicePointer()) read and write, and that streams wait for (see streamWaitMemory()). Own
   * streams (HIP, see enableOwnQueues()) wait for it in the command processor of the device, as
   * for a signal of HSA, instead of with a kernel that polls it.
   */
  virtual uint32_t* allocCounter() {
    auto* counter =
        static_cast<uint32_t*>(allocPinnedMem(sizeof(uint32_t), false, Destination::CurrentDevice));
    *counter = 0;
    return counter;
  }
  virtual void freeCounter(uint32_t* counter) { freePinnedMem(counter); }

  /**
   * Traces the work of the device piece by piece (HIP with own queues: each kernel and each wait of
   * an own queue): calls `sink` with its name, the number of its queue, and its begin and end on
   * the clock of the host, from a thread of the library, some time after it ran. Returns whether
   * the device traces. An empty sink stops tracing, once the work traced so far has been reported.
   * Tracing costs the device a little for each kernel.
   */
  using DeviceTraceSink = std::function<void(std::string_view name,
                                             int queue,
                                             std::chrono::steady_clock::time_point begin,
                                             std::chrono::steady_clock::time_point end)>;
  virtual bool traceDevice(DeviceTraceSink /*sink*/) { return false; }

  virtual void* createEvent(bool withTiming = false) = 0;
  virtual void destroyEvent(void* eventPtr) = 0;
  virtual void syncEventWithHost(void* eventPtr) = 0;
  virtual bool isEventCompleted(void* eventPtr) = 0;
  virtual void recordEventOnHost(void* eventPtr) = 0;
  virtual void recordEventOnStream(void* eventPtr, void* streamPtr) = 0;
  virtual double timespanEvents(void* eventPtrStart, void* eventPtrEnd) = 0;

  /**
   * Whether timespanEvents() can time events that were created with timing and recorded on
   * streams. The SYCL backend needs either profiling on its queues (ENABLE_PROFILING_MARKERS=ON),
   * or a device with profiling tags (sycl_ext_oneapi_profiling_tag) for that.
   */
  virtual bool isCapableOfEventTiming() = 0;

  virtual bool isUnifiedMemoryDefault() = 0;

  virtual void initialize() = 0;
  virtual void finalize() = 0;
  virtual void profilingMessage(const std::string& message) = 0;
  virtual void putProfilingMark(const std::string& name, ProfilingColors color) = 0;
  virtual void popLastProfilingMark() = 0;

  virtual void setupPrinting(int rank) = 0;

  bool hasFinalized() { return m_isFinalized; }

  protected:
  bool m_isFinalized = false;
  std::mutex apiMutex;
};
} // namespace device

#endif // SEISSOLDEVICE_ABSTRACTAPI_H_
