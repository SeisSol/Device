// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HIP_OWNQUEUES_H_
#define SEISSOLDEVICE_INTERFACES_HIP_OWNQUEUES_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <memory>

/**
 * Streams on HSA queues of our own.
 *
 * HIP (CLR) puts all streams onto a few hardware queues that they share in turn (GPU_MAX_HW_QUEUES
 * per priority, 4 by default), plus one more stream for each branch of each graph: a stream that
 * waits holds up the streams behind it on its queue, and a wait for an event of another queue costs
 * ~12 us (MI250X, ROCm 6.3). A GCD has 24 hardware queues only; with more, the queues that wait may
 * keep the ones they wait for from getting a slot.
 *
 * An own stream runs its work on an HSA queue that only the own streams with the same queue number
 * share, in the order it got enqueued. Kernels become AQL dispatch packets that we write ourselves:
 * the kernel object comes from the loader, the layout of the arguments from the metadata of the
 * code object (with the hidden arguments filled in as HIP does). A wait for another queue is a
 * barrier packet on the signal that the marks of that queue count down (a mark completes once all
 * work before it on its queue has). Copies and fills are kernels of our own; a host function runs
 * on a thread of our own, while its queue waits for it.
 *
 * The handle of an own stream is a tagged pointer (bit 0 set), which no HIP call must get: kernel
 * launches reach launch() through hipLaunchKernel, which the program defines itself (see
 * OwnQueueHook.cpp). While an own stream gets captured into a graph, all of its work goes to a
 * stream of HIP that stands in for it (its vehicle). A graph that gets launched on an own stream
 * gets translated into packets, once per graph and queue; its temporary allocations (hipMallocAsync
 * while capturing) get memory of the queue instead, which the translation puts into the arguments.
 *
 * Events are tagged pointers as well, once own queues are enabled: recorded on an own stream, an
 * event is a mark of its queue; recorded on a stream of HIP, an event of HIP. With
 * DEVICE_OWN_QUEUE_TIMING=1, events with timing take the time by a kernel each.
 *
 * Fences: a mark writes back the L2 cache (release of the system), for the host and for MPI, which
 * read device memory past it. Kernels and waits for other own queues only fence for the device
 * (agent): an acquire of the system in a packet drops the whole L2 cache, at the expense of the
 * kernels of all queues. Where the work has to see what the host wrote (after a wait for a counter
 * or a host function, and in copies from host memory), a kernel of our own fences in the shader
 * instead, which drops only the lines that may be stale (DEVICE_OWN_QUEUE_HOST_ACQUIRE=packet and
 * DEVICE_OWN_QUEUE_WAIT_SCOPE=system go back to fences of packets).
 */
namespace device::ownqueues {

inline bool isTagged(const void* handle) {
  return (reinterpret_cast<std::uintptr_t>(handle) & 1U) != 0;
}

/// the launch of HIP, which the hook in the program hands on to for streams of HIP
using LaunchFunction = hipError_t (*)(const void*, dim3, dim3, void**, std::size_t, hipStream_t);

/// called by the hook in the program (see OwnQueueHook.cpp), at its start
void registerLaunchHook(LaunchFunction hipLaunch);

/// enables own queues, if HSA and the launch hook are there; returns whether they are enabled
bool enable();
bool enabled();

/// a stream on the own queue with the given number (created on demand); a priority (0 to 1, unless
/// NAN) applies to the whole queue
void* createStream(int queue, double priority);
void destroyStream(void* stream);

/// the stream of HIP that stands in for an own stream while it gets captured; HIP streams stay
hipStream_t vehicle(void* stream);

/// whether the own stream gets captured into a graph right now (see beginCapture())
bool capturing(void* stream);
void beginCapture(void* stream);
void endCapture(void* stream);

/// kernel launches (from the hook in the program)
hipError_t launch(const void* function,
                  dim3 grid,
                  dim3 block,
                  void** args,
                  std::size_t sharedMemBytes,
                  hipStream_t stream);

void syncStream(void* stream);
bool streamDone(void* stream);
void hostFunction(void* stream, const std::function<void()>& function);
void waitMemory(void* stream, std::uint32_t* location, std::uint32_t value);

/// counters for streamWaitMemory() that the command processor waits for: the values of signals
std::uint32_t* allocCounter();
void freeCounter(std::uint32_t* counter);
/// whether the memory is a counter from allocCounter()
bool isCounter(const void* location);

/// copies between any memory the device can access; pageable host memory synchronously
void copy(void* dst, const void* src, std::size_t bytes, void* stream);

void* allocAsync(std::size_t bytes, void* stream);
void freeAsync(void* ptr, void* stream);

void* createEvent(bool withTiming);
void destroyEvent(void* event);
/// the event of HIP of an own event (for streams of HIP); HIP events stay
hipEvent_t hipEventOf(void* event);
void recordEvent(void* event, void* stream);
void recordEventOnHost(void* event);
void streamWaitEvent(void* stream, void* event);
/// the time between two events, in seconds: by the kernels that took it for own events with timing
double timespan(void* start, void* end);
/// whether events with timing take the time on own streams (DEVICE_OWN_QUEUE_TIMING=1)
bool timing();
bool eventDone(void* event);
void syncEvent(void* event);

/// the graph of HIP that gets translated (see Program); one per DeviceGraph
class Program;
struct Programs {
  Programs();
  ~Programs();
  Programs(const Programs&) = delete;
  Programs& operator=(const Programs&) = delete;
  std::unique_ptr<struct ProgramsImpl> impl;
};
void launchGraph(Programs& programs, hipGraph_t graph, void* stream);

/// waits until all own queues are done
void syncAll();

void finalize();

} // namespace device::ownqueues

#endif // SEISSOLDEVICE_INTERFACES_HIP_OWNQUEUES_H_
