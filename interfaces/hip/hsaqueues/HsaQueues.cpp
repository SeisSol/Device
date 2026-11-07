// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "HsaQueues.h"

#include "Common.h"
#include "Internals.h"
#include "Kernels.h"
#include "Program.h"
#include "Queue.h"
#include "Runtime.h"
#include "Trace.h"
#include "utils/logger.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <hip/hip_runtime.h>
#include <hsa/amd_hsa_queue.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace device::hsaqueues {

namespace {

// ------------------------------------------------------------------------------------------------
// streams and events

struct Stream {
  Queue* queue{nullptr};
  // for the DAG log
  std::string name;
  // stands in for the stream while it gets captured
  hipStream_t vehicle{nullptr};
  int captures{0};
  // the write index of the queue after the latest work of the stream
  std::uint64_t lastIndex{0};
};

// what an event stands for since it got recorded last
struct EventState {
  // recorded on an HSA stream: the mark of the queue
  Queue* queue{nullptr};
  std::uint64_t mark{0};
  // with timing, on an HSA stream: whether a kernel took the time there (into Event::stamp)
  bool stamped{false};
  // recorded on a stream of HIP (see Runtime::hipWaitsOnDevice): the counter of that stream (a
  // signal that only counts up, with the address of its value) and the value it gets after the work
  // before the event
  hsa_signal_t hipCounter{};
  const std::int64_t* hipCounterValue{nullptr};
  std::int64_t hipValue{0};
};

struct Event {
  hipEvent_t hip{nullptr};
  // with timing (on HSA streams): the signal of the kernel that takes the time
  bool timing{false};
  hsa_signal_t stamp{};

  /// the state of the event as of its latest record: records and waits may come from different
  /// threads (the mutex of a queue never gets taken while this lock is held)
  EventState state() {
    const std::lock_guard lock(mutex_);
    return state_;
  }
  void record(const EventState& state) {
    const std::lock_guard lock(mutex_);
    state_ = state;
  }

  private:
  std::mutex mutex_;
  EventState state_;
};

Stream* streamOf(void* handle) {
  return reinterpret_cast<Stream*>(reinterpret_cast<std::uintptr_t>(handle) & ~std::uintptr_t{1});
}

Event* eventOf(void* handle) {
  return reinterpret_cast<Event*>(reinterpret_cast<std::uintptr_t>(handle) & ~std::uintptr_t{1});
}

template <typename T>
void* tag(T* pointer) {
  return reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(pointer) | 1U);
}

std::atomic<bool> enabledFlag{false};
LaunchFunction hipLaunch{nullptr};

/// memory that is not that of the device (of the host, or of other agents): its lines in the L2
/// cache may be stale, and the others read it past that cache
bool foreignMemory(const void* ptr) {
  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  return hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS ||
         info.type != HSA_EXT_POINTER_TYPE_HSA ||
         info.agentOwner.handle != Runtime::instance().gpu.handle;
}

void dispatchCopy(Queue& queue, void* dst, const void* src, std::size_t bytes) {
  const auto blocks = static_cast<std::uint32_t>(
      std::min<std::size_t>(MaxCopyBlocks, std::max<std::size_t>(1, bytes / (16 * CopyThreads))));
  void* args[] = {&dst, const_cast<void**>(&src), &bytes};
  // sees what others wrote into their memory, and writes back what it wrote into theirs
  const bool fromForeign = foreignMemory(src);
  if (fromForeign) {
    queue.acquireFromHost();
  }
  const auto acquire = fromForeign && !Runtime::instance().acquireByKernel ? HSA_FENCE_SCOPE_SYSTEM
                                                                           : HSA_FENCE_SCOPE_AGENT;
  const auto release = foreignMemory(dst) ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_AGENT;
  queue.dispatch(Queue::kernelOf(reinterpret_cast<const void*>(hsaCopy)),
                 dim3(blocks),
                 dim3(CopyThreads),
                 0,
                 args,
                 fenceBits(acquire, release),
                 hsa_signal_t{0});
}

} // namespace

// ------------------------------------------------------------------------------------------------
// the interface

namespace {
/// other work on a stream for the DAG log (see DagLog)
void logOther(const Stream* stream, Queue& queue, std::uint64_t first, const char* kind) {
  if (Runtime::instance().schedule) {
    queue.predictWork(steadySeconds(),
                      std::string_view(kind) == "copy" ? DefaultCopySeconds : DefaultHostSeconds);
  }
  if (dagLog().on()) {
    dagLog().line("O %lld %p %d %llu %llu %s\n",
                  DagLog::now(),
                  static_cast<const void*>(stream),
                  queue.number(),
                  static_cast<unsigned long long>(first),
                  static_cast<unsigned long long>(stream->lastIndex),
                  kind);
  }
}
} // namespace

void registerLaunchHook(LaunchFunction launch) { hipLaunch = launch; }

bool enable() {
  if (enabledFlag.load()) {
    return true;
  }
  if (hipLaunch == nullptr) {
    logWarning() << "HSA queues need the launch hook in the program (Hook.cpp).";
    return false;
  }
  if (!Runtime::instance().setUp()) {
    return false;
  }
  enabledFlag.store(true);
  return true;
}

bool enabled() { return enabledFlag.load(std::memory_order_relaxed); }

void* createStream(int queue, double priority) {
  auto* stream = new Stream();
  stream->queue = &Runtime::instance().queue(queue);
  if (!std::isnan(priority)) {
    const auto level = priority < 1.0 / 3   ? HSA_AMD_QUEUE_PRIORITY_LOW
                       : priority > 2.0 / 3 ? HSA_AMD_QUEUE_PRIORITY_HIGH
                                            : HSA_AMD_QUEUE_PRIORITY_NORMAL;
    check(hsa_amd_queue_set_priority(stream->queue->hsaQueue(), level),
          "hsa_amd_queue_set_priority");
  }
  APIWRAP(hipStreamCreateWithFlags(&stream->vehicle, hipStreamNonBlocking));
  return tag(stream);
}

void destroyStream(void* handle) {
  auto* stream = streamOf(handle);
  syncStream(handle);
  APIWRAP(hipStreamDestroy(stream->vehicle));
  delete stream;
}

hipStream_t vehicle(void* handle) {
  if (!isTagged(handle)) {
    return static_cast<hipStream_t>(handle);
  }
  return streamOf(handle)->vehicle;
}

bool capturing(void* handle) { return isTagged(handle) && streamOf(handle)->captures > 0; }

void beginCapture(void* handle) {
  if (isTagged(handle)) {
    // (a capture only records; the graph runs in the order of the stream once it gets launched)
    ++streamOf(handle)->captures;
  }
}

void endCapture(void* handle) {
  if (isTagged(handle)) {
    --streamOf(handle)->captures;
  }
}

hipError_t launch(const void* function,
                  dim3 grid,
                  dim3 block,
                  void** args,
                  std::size_t sharedMemBytes,
                  hipStream_t handle) {
  auto* stream = streamOf(handle);
  if (stream->captures > 0) {
    return hipLaunch(function, grid, block, args, sharedMemBytes, stream->vehicle);
  }
  if (grid.x == 0 || grid.y == 0 || grid.z == 0 || block.x == 0 || block.y == 0 || block.z == 0) {
    // (as HIP: nothing gets launched)
    return hipErrorInvalidConfiguration;
  }
  const auto& kernel = Runtime::instance().kernel(function);
  auto& queue = *stream->queue;
  const std::lock_guard lock(queue.mutex);
  const auto first = queue.created();
  queue.dispatch(kernel, grid, block, sharedMemBytes, args, KernelFences, hsa_signal_t{0});
  stream->lastIndex = queue.created();
  if (Runtime::instance().schedule) {
    const auto estimate = Runtime::instance().kernelEstimate(kernel.object);
    queue.predictWork(steadySeconds(), estimate > 0 ? estimate : DefaultKernelSeconds);
  }
  if (dagLog().on()) {
    dagLog().line("L %lld %p %d %llu %llu %s\n",
                  DagLog::now(),
                  static_cast<void*>(stream),
                  queue.number(),
                  static_cast<unsigned long long>(first),
                  static_cast<unsigned long long>(stream->lastIndex),
                  kernel.name.c_str());
  }
  return hipSuccess;
}

void syncStream(void* handle) {
  auto* stream = streamOf(handle);
  auto& queue = *stream->queue;
  std::uint64_t mark = 0;
  {
    const std::lock_guard lock(queue.mutex);
    mark = queue.markCoveringLocked(stream->lastIndex);
  }
  queue.wait(mark);
  const std::lock_guard lock(queue.mutex);
  queue.reclaim();
}

bool streamDone(void* handle) {
  auto* stream = streamOf(handle);
  auto& queue = *stream->queue;
  const std::lock_guard lock(queue.mutex);
  return queue.reached(queue.markCoveringLocked(stream->lastIndex));
}

void hostFunction(void* handle, const std::function<void()>& function) {
  auto* stream = streamOf(handle);
  auto& queue = *stream->queue;
  const std::lock_guard lock(queue.mutex);
  const auto first = queue.created();
  queue.hostFunction(function);
  stream->lastIndex = queue.created();
  logOther(stream, queue, first, "host");
}

namespace {
std::mutex counterMutex;
std::unordered_map<const void*, hsa_signal_t> counterSignals;

hsa_signal_t counterSignal(const void* location) {
  const std::lock_guard lock(counterMutex);
  const auto found = counterSignals.find(location);
  return found == counterSignals.end() ? hsa_signal_t{0} : found->second;
}
} // namespace

std::uint32_t* allocCounter() {
  // a signal that only the device waits for (the host polls it), whose value the host stores into
  // as into memory: the command processor checks it
  hsa_signal_t signal{};
  check(hsa_amd_signal_create(0, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal),
        "hsa_amd_signal_create");
  volatile hsa_signal_value_t* value = nullptr;
  check(hsa_amd_signal_value_pointer(signal, &value), "hsa_amd_signal_value_pointer");
  // (the lower half of the 64-bit value, which the higher one leaves at 0)
  auto* counter = reinterpret_cast<std::uint32_t*>(const_cast<hsa_signal_value_t*>(value));
  const std::lock_guard lock(counterMutex);
  counterSignals.emplace(counter, signal);
  return counter;
}

void freeCounter(std::uint32_t* counter) {
  hsa_signal_t signal{};
  {
    const std::lock_guard lock(counterMutex);
    const auto found = counterSignals.find(counter);
    if (found == counterSignals.end()) {
      return;
    }
    signal = found->second;
    counterSignals.erase(found);
  }
  hsa_signal_destroy(signal);
}

bool isCounter(const void* location) {
  if (location == nullptr) {
    return false;
  }
  const std::lock_guard lock(counterMutex);
  return counterSignals.find(location) != counterSignals.end();
}

void waitMemory(void* handle, std::uint32_t* location, std::uint32_t value) {
  auto* stream = streamOf(handle);
  auto& queue = *stream->queue;
  const auto signal = counterSignal(location);
  if (signal.handle != 0 && Runtime::instance().barrierValue) {
    // the command processor waits for the signal
    const std::lock_guard lock(queue.mutex);
    const auto first = queue.created();
    queue.waitForCounter(signal, value, 0xffffffff);
    stream->lastIndex = queue.created();
    logOther(stream, queue, first, "counter");
    return;
  }
  std::uint32_t* deviceLocation = location;
  if (signal.handle == 0) {
    APIWRAP(hipHostGetDevicePointer(reinterpret_cast<void**>(&deviceLocation), location, 0));
  }
  const std::uint32_t* counter = deviceLocation;
  void* args[] = {&counter, &value};
  const std::lock_guard lock(queue.mutex);
  const auto first = queue.created();
  queue.dispatch(Queue::kernelOf(reinterpret_cast<const void*>(hsaWaitHost)),
                 dim3(1),
                 dim3(1),
                 0,
                 args,
                 queue.hostWaitFences(),
                 hsa_signal_t{0});
  queue.acquireFromHost();
  stream->lastIndex = queue.created();
  logOther(stream, queue, first, "counter");
}

void copy(void* dst, const void* src, std::size_t bytes, void* handle) {
  if (bytes == 0) {
    return;
  }
  auto* stream = streamOf(handle);
  auto& queue = *stream->queue;
  const void* srcDevice = deviceAddress(src);
  void* dstDevice = const_cast<void*>(deviceAddress(dst));
  if (srcDevice == nullptr && dstDevice != nullptr &&
      bytes <= Runtime::instance().kernargRingBytes / 64) {
    // from pageable host memory, as HIP does: staged right away, here in the memory of the kernel
    // arguments, which stays until the copy is done
    const std::lock_guard lock(queue.mutex);
    auto* staging = static_cast<char*>(queue.kernargs(bytes, 16));
    std::memcpy(staging, src, bytes);
    flushKernargs(staging + bytes - 1);
    const auto first = queue.created();
    dispatchCopy(queue, dstDevice, staging, bytes);
    stream->lastIndex = queue.created();
    logOther(stream, queue, first, "copy");
    return;
  }
  if (srcDevice == nullptr || dstDevice == nullptr) {
    // pageable host memory: once the stream is done, synchronously
    warnOnce("copies from or to pageable host memory wait for their stream");
    syncStream(handle);
    APIWRAP(hipMemcpy(dst, src, bytes, hipMemcpyDefault));
    return;
  }
  const std::lock_guard lock(queue.mutex);
  const auto first = queue.created();
  dispatchCopy(queue, dstDevice, srcDevice, bytes);
  stream->lastIndex = queue.created();
  logOther(stream, queue, first, "copy");
}

void* allocAsync(std::size_t bytes, void* handle) {
  auto& queue = *streamOf(handle)->queue;
  const std::lock_guard lock(queue.mutex);
  return queue.allocAsync(bytes);
}

void freeAsync(void* ptr, void* handle) {
  auto* stream = streamOf(handle);
  auto& queue = *stream->queue;
  const std::lock_guard lock(queue.mutex);
  queue.freeAsync(ptr, stream->lastIndex);
}

namespace {
// the counters of the streams of HIP that events got recorded on (see Runtime::hipWaitsOnDevice),
// with the value of the latest event; the lock keeps the values in the order of the streams. Their
// signals count in 64 bits, as a mark does (a counter of the host has 32 bits only).
struct HipStreamCounter {
  hsa_signal_t signal{};
  const std::int64_t* value{nullptr};
  std::int64_t count{0};
};
std::mutex hipCounterMutex;
std::unordered_map<hipStream_t, HipStreamCounter> hipCounters;

/// lets a stream of HIP wait until mark `mark` of `queue` has been reached: by a kernel of ours
/// there that waits for the counter of the marks
void waitOnHipStream(const Queue& queue, std::uint64_t mark, hipStream_t stream) {
  if (queue.reached(mark)) {
    return;
  }
  const std::int64_t* value = queue.counterValue();
  std::int64_t below = CounterStart - static_cast<std::int64_t>(mark) + 1;
  void* args[] = {&value, &below};
  APIWRAP(
      hipLaunch(reinterpret_cast<const void*>(hsaHipWaitBelow), dim3(1), dim3(1), args, 0, stream));
}

/// lets an HSA stream wait for an event recorded on a stream of HIP: until the counter of that
/// stream has reached the value of the event (see recordEvent())
void waitHipCounter(Stream& stream, const EventState& state) {
  auto& queue = *stream.queue;
  const std::lock_guard lock(queue.mutex);
  const auto first = queue.created();
  if (Runtime::instance().barrierValue) {
    queue.waitForCounter(state.hipCounter, state.hipValue, -1);
  } else {
    const std::int64_t* value = state.hipCounterValue;
    std::int64_t atLeast = state.hipValue;
    void* args[] = {&value, &atLeast};
    queue.dispatch(Queue::kernelOf(reinterpret_cast<const void*>(hsaWaitAtLeast)),
                   dim3(1),
                   dim3(1),
                   0,
                   args,
                   queue.hostWaitFences(),
                   hsa_signal_t{0});
    queue.acquireFromHost();
  }
  stream.lastIndex = queue.created();
  logOther(&stream, queue, first, "counter");
}
} // namespace

void* createEvent(bool withTiming) {
  auto* event = new Event();
  event->timing = withTiming && Runtime::instance().timing;
  if (event->timing) {
    check(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &event->stamp),
          "hsa_amd_signal_create");
  }
  if (withTiming) {
    APIWRAP(hipEventCreate(&event->hip));
  } else {
    APIWRAP(hipEventCreateWithFlags(&event->hip, hipEventDisableTiming));
  }
  return tag(event);
}

void destroyEvent(void* handle) {
  auto* event = eventOf(handle);
  if (event->stamp.handle != 0) {
    hsa_signal_destroy(event->stamp);
  }
  internals::forgetEvent(event->hip);
  APIWRAP(hipEventDestroy(event->hip));
  delete event;
}

hipEvent_t hipEventOf(void* handle) {
  return isTagged(handle) ? eventOf(handle)->hip : static_cast<hipEvent_t>(handle);
}

void recordEvent(void* eventHandle, void* streamHandle) {
  auto* event = eventOf(eventHandle);
  EventState state;
  if (isTagged(streamHandle) && !capturing(streamHandle)) {
    auto* stream = streamOf(streamHandle);
    auto& queue = *stream->queue;
    const std::lock_guard lock(queue.mutex);
    if (event->timing) {
      queue.stamp(event->stamp);
      stream->lastIndex = queue.created();
      state.stamped = true;
    }
    state.queue = &queue;
    state.mark = queue.markCoveringLocked(stream->lastIndex);
    event->record(state);
    if (dagLog().on()) {
      dagLog().line("R %lld %p %d %p %llu %llu\n",
                    DagLog::now(),
                    static_cast<void*>(stream),
                    queue.number(),
                    static_cast<void*>(event),
                    static_cast<unsigned long long>(state.mark),
                    static_cast<unsigned long long>(stream->lastIndex));
    }
    return;
  }
  // on a stream of HIP, or one that gets captured
  auto hipStream = vehicle(streamHandle);
  internals::anchorCapture(hipStream);
  if (Runtime::instance().hipWaitsOnDevice && !isTagged(streamHandle) &&
      internals::captureState(hipStream).status != hipStreamCaptureStatusActive) {
    // HSA streams wait for the event by a counter that a kernel of ours sets after it (CLR does not
    // give the signal of an event)
    const std::lock_guard lock(hipCounterMutex);
    auto& counter = hipCounters[hipStream];
    if (counter.signal.handle == 0) {
      check(hsa_amd_signal_create(0, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &counter.signal),
            "hsa_amd_signal_create");
      volatile hsa_signal_value_t* value = nullptr;
      check(hsa_amd_signal_value_pointer(counter.signal, &value), "hsa_amd_signal_value_pointer");
      counter.value = const_cast<const std::int64_t*>(value);
    }
    APIWRAP(hipEventRecord(event->hip, hipStream));
    internals::noteRecordedEvent(event->hip, hipStream);
    auto* location = const_cast<std::int64_t*>(counter.value);
    std::int64_t value = ++counter.count;
    void* args[] = {&location, &value};
    APIWRAP(hipLaunch(
        reinterpret_cast<const void*>(hsaWriteCounter), dim3(1), dim3(1), args, 0, hipStream));
    state.hipCounter = counter.signal;
    state.hipCounterValue = counter.value;
    state.hipValue = value;
    event->record(state);
    return;
  }
  APIWRAP(hipEventRecord(event->hip, hipStream));
  internals::noteRecordedEvent(event->hip, hipStream);
  event->record(state);
}

double timespan(void* startHandle, void* endHandle) {
  auto* start = eventOf(startHandle);
  auto* end = eventOf(endHandle);
  const auto startState = start->state();
  const auto endState = end->state();
  if (startState.stamped && endState.stamped) {
    const auto stamp = [](hsa_signal_t signal) {
      while (hsa_signal_wait_scacquire(
                 signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) >= 1) {
      }
      hsa_amd_profiling_dispatch_time_t time{};
      check(hsa_amd_profiling_get_dispatch_time(Runtime::instance().gpu, signal, &time),
            "hsa_amd_profiling_get_dispatch_time");
      return time.end;
    };
    const auto first = stamp(start->stamp);
    const auto second = stamp(end->stamp);
    // a stamp without a time (e.g. of a dispatch without profiling) translates to anything
    std::uint64_t now = 0;
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &now);
    const auto valid = [&](std::uint64_t time) {
      return time >= Runtime::instance().startTimestamp && time <= now;
    };
    if (!valid(first) || !valid(second)) {
      warnOnce("events without times");
      return 0;
    }
    const auto ticks = static_cast<double>(second) - static_cast<double>(first);
    return ticks / static_cast<double>(Runtime::instance().timestampFrequency);
  }
  if (startState.queue == nullptr && endState.queue == nullptr && !startState.stamped &&
      !endState.stamped) {
    float time{};
    APIWRAP(hipEventElapsedTime(&time, start->hip, end->hip));
    return static_cast<double>(time) / 1000.0;
  }
  return 0;
}

bool timing() { return Runtime::instance().timing; }

void nameStream(void* handle, const std::string& name) {
  if (!isTagged(handle)) {
    return;
  }
  auto* stream = streamOf(handle);
  stream->name = name;
  if (dagLog().on()) {
    dagLog().line("N %p %s\n", static_cast<void*>(stream), name.c_str());
  }
}

bool trace(const TraceSink& sink) {
  if (!enabled()) {
    return false;
  }
  if (sink) {
    tracer().start(sink);
  } else {
    tracer().stop();
    dagLog().flush();
  }
  return true;
}

void recordEventOnHost(void* handle) {
  auto* event = eventOf(handle);
  event->record(EventState{});
  internals::forgetEvent(event->hip);
  APIWRAP(hipEventRecord(event->hip));
}

void streamWaitEvent(void* streamHandle, void* eventHandle) {
  if (isTagged(streamHandle) && !capturing(streamHandle)) {
    auto* stream = streamOf(streamHandle);
    if (!isTagged(eventHandle)) {
      warnOnce("HSA streams wait for events of HIP on the host");
      APIWRAP(hipEventSynchronize(static_cast<hipEvent_t>(eventHandle)));
      return;
    }
    auto* event = eventOf(eventHandle);
    const auto state = event->state();
    if (state.queue == nullptr) {
      if (state.hipCounter.handle != 0) {
        // the counter that a kernel of ours sets after the event on the stream of HIP
        waitHipCounter(*stream, state);
        return;
      }
      warnOnce("HSA streams wait for events recorded on streams of HIP on the host");
      APIWRAP(hipEventSynchronize(event->hip));
      return;
    }
    auto& queue = *stream->queue;
    const std::lock_guard lock(queue.mutex);
    queue.waitFor(*state.queue, state.mark);
    stream->lastIndex = queue.created();
    if (dagLog().on()) {
      dagLog().line("W %lld %p %d %p\n",
                    DagLog::now(),
                    static_cast<void*>(stream),
                    queue.number(),
                    static_cast<void*>(event));
    }
    return;
  }
  auto hipStream = vehicle(streamHandle);
  if (const auto state = isTagged(eventHandle) ? eventOf(eventHandle)->state() : EventState{};
      state.queue != nullptr) {
    if (internals::captureState(hipStream).status == hipStreamCaptureStatusActive) {
      logError() << "HSA queues: a stream that gets captured cannot wait for an HSA stream.";
    }
    if (Runtime::instance().hipWaitsOnDevice) {
      waitOnHipStream(*state.queue, state.mark, hipStream);
      return;
    }
    warnOnce("streams of HIP wait for events of HSA streams on the host");
    state.queue->wait(state.mark);
    return;
  }
  const auto hipEvent = hipEventOf(eventHandle);
  if (!internals::dependOnCapturedEvent(hipStream, hipEvent)) {
    APIWRAP(hipStreamWaitEvent(hipStream, hipEvent, 0));
  }
}

bool eventDone(void* handle) {
  auto* event = eventOf(handle);
  if (const auto state = event->state(); state.queue != nullptr) {
    return state.queue->reached(state.mark);
  }
  return APIWRAPX(hipEventQuery(event->hip), {hipErrorNotReady}) == hipSuccess;
}

void syncEvent(void* handle) {
  auto* event = eventOf(handle);
  if (const auto state = event->state(); state.queue != nullptr) {
    state.queue->wait(state.mark);
    return;
  }
  APIWRAP(hipEventSynchronize(event->hip));
}

void launchGraph(Programs& programs, hipGraph_t graph, void* handle) {
  auto* stream = streamOf(handle);
  auto& queue = *stream->queue;
  Program* program = nullptr;
  {
    const std::lock_guard lock(programs.impl->mutex);
    auto& entry = programs.impl->byQueue[&queue];
    if (entry == nullptr) {
      entry = std::make_unique<Program>(queue, graph, programs.impl->origins);
    }
    program = entry.get();
  }
  const std::lock_guard lock(queue.mutex);
  const auto first = queue.created();
  stream->lastIndex = program->submit();
  if (dagLog().on()) {
    if (dagLog().firstTime(program)) {
      dagLog().line(
          "P %p %d%s\n", static_cast<void*>(program), queue.number(), program->levels().c_str());
    }
    dagLog().line("G %lld %p %d %llu %llu %p\n",
                  DagLog::now(),
                  static_cast<void*>(stream),
                  queue.number(),
                  static_cast<unsigned long long>(first),
                  static_cast<unsigned long long>(stream->lastIndex),
                  static_cast<void*>(program));
  }
}

bool noteOrigin(Programs& programs, hipGraphNode_t node, void* handle) {
  auto* stream = streamOf(handle);
  const std::lock_guard lock(programs.impl->mutex);
  return programs.impl->origins.emplace(node, stream->queue->number()).second;
}

void syncAll() {
  if (!enabled()) {
    return;
  }
  for (auto* queue : Runtime::instance().allQueues()) {
    std::uint64_t mark = 0;
    {
      const std::lock_guard lock(queue->mutex);
      mark = queue->markCoveringLocked(queue->created());
    }
    queue->wait(mark);
  }
}

void finalize() {
  if (!enabled()) {
    return;
  }
  syncAll();
  Runtime::instance().finalize();
  enabledFlag.store(false);
}

} // namespace device::hsaqueues
