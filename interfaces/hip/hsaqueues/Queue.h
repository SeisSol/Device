// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_QUEUE_H_
#define SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_QUEUE_H_

#include "Common.h"
#include "Internals.h"
#include "Kernels.h"
#include "Runtime.h"
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

/**
 * An HSA queue of our own. Its marks (barrier packets that wait for all work before them) count its
 * counter signal down, from CounterStart; mark m has been reached once the counter is at most
 * CounterStart - m. The arguments of the kernels come from a ring, which marks free again.
 *
 * The host never waits to enqueue: the queue may wait for work that the host has yet to enqueue
 * (e.g. for halo data that MPI receives on the host). Packets that find the queue full wait in host
 * memory until the host thread (or the next packet) writes them into it; arguments that find the
 * ring full get memory of their own, which the next mark frees again.
 *
 * Where the command processor does not wait for values (Runtime::markSignals), each mark counts
 * down a signal of its own as well, just before the counter; other queues wait for it by
 * barrier-AND packets. The signal goes back to the pool once the mark has been reached and each
 * queue that waits for it has reached its next mark after the packet: only then has that packet
 * surely done (the read index of a queue does not tell, as a command processor may take packets
 * ahead, as on RDNA).
 */
class Queue {
  public:
  /// on the HSA queue `placed` (see Runtime::placeQueue()), or on a new one
  Queue(Runtime& runtime, int number, hsa_queue_t* placed = nullptr)
      : runtime_(runtime), number_(number), queue_(placed) {
    if (queue_ == nullptr) {
      queue_ = createHsaQueue();
    }
    check(hsa_amd_signal_create(CounterStart, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &counter_),
          "hsa_amd_signal_create");
    check(hsa_amd_signal_value_pointer(counter_, &counterValue_), "hsa_amd_signal_value_pointer");
    if (runtime.timing || runtime.tracing.load() || runtime.schedule) {
      // before the first packet: enabled later, the first stamps of the queue had no times
      enableProfiling();
    }
    ringBytes_ = runtime.kernargRingBytes;
    ring_ = static_cast<char*>(allocKernargMemory(ringBytes_));
    APIWRAP(hipStreamCreateWithFlags(&helper_, hipStreamNonBlocking));
    if (setting<bool>("VERBOSE", false)) {
      logInfo() << "HSA queue" << number << ": packets at" << queue_->base_address
                << ", marks count" << reinterpret_cast<void*>(counter_.handle);
    }
  }

  ~Queue() {
    // (the host thread has written all packets by now, see finalize())
    const auto last = markCoveringLocked(created_);
    wait(last);
    reclaim();
    {
      // (queues that wait for them may go later: their signals stay until all have gone)
      const std::lock_guard lock(signalMutex_);
      for (const auto& entry : markSignals_) {
        runtime_.retireMarkSignal(entry.signal);
      }
      markSignals_.clear();
    }
    for (const auto& [arena, bytes] : arenas_) {
      static_cast<void>(hipFree(arena));
      runtime_.deviceBytes -= bytes;
    }
    static_cast<void>(hipStreamDestroy(helper_));
    freeKernargMemory(ring_, ringBytes_);
    hsa_signal_destroy(counter_);
    hsa_queue_destroy(queue_);
  }

  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;

  [[nodiscard]] int number() const { return number_; }
  [[nodiscard]] hsa_queue_t* hsaQueue() const { return queue_; }

  // all of the following but reached() and wait() with `mutex` held

  /// enqueues a packet: into the queue, or into host memory while it is full (see drain())
  void push(const PacketImage& packet) {
    const bool tracing = runtime_.tracing.load(std::memory_order_relaxed);
    if (!tracing && !runtime_.extDispatch) {
      pushImage(packet);
      return;
    }
    PacketImage image = packet;
    if (runtime_.extDispatch) {
      // (a wait that it does not take comes first)
      extend(image);
    }
    if (tracing) {
      annotate(image);
    }
    pushImage(image);
  }

  /// a kernel or wait whose times the trace takes, by a signal of its own (see trace())
  struct TraceEntry {
    hsa_signal_t signal{};
    // the number of the packet in the queue (see created())
    std::uint64_t index{0};
    std::uint64_t object{0};
    // the queue a wait waits for, or none: a wait for the host
    std::optional<int> waitsFor;
    bool kernel{true};
  };

  /// moves the entries of the trace whose packets have completed into `done`, in their order
  void takeTraced(std::vector<TraceEntry>& done) {
    const std::lock_guard lock(traceMutex_);
    while (!traced_.empty() && hsa_signal_load_relaxed(traced_.front().signal) < 1) {
      done.push_back(traced_.front());
      traced_.pop_front();
    }
  }

  [[nodiscard]] bool traceEmpty() {
    const std::lock_guard lock(traceMutex_);
    return traced_.empty();
  }

  /// lets the command processor take the times of the packets from now on
  void enableProfiling() {
    if (!profiling_) {
      check(hsa_amd_profiling_set_profiler_enabled(queue_, 1),
            "hsa_amd_profiling_set_profiler_enabled");
      profiling_ = true;
    }
  }

  [[nodiscard]] std::uint64_t counterHandle() const { return counter_.handle; }
  /// the value of the counter of the marks (see waitFor()), for kernels that wait for it
  [[nodiscard]] const std::int64_t* counterValue() const {
    return const_cast<const std::int64_t*>(counterValue_);
  }

  private:
  /// gives kernels and waits without a completion signal one that takes their times
  void annotate(PacketImage& image) {
    std::uint32_t word = 0;
    std::memcpy(&word, image.bytes.data(), sizeof(word));
    const auto type =
        (word >> HSA_PACKET_HEADER_TYPE) & ((1U << HSA_PACKET_HEADER_WIDTH_TYPE) - 1U);
    TraceEntry entry;
    hsa_signal_t* completion = nullptr;
    if (type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      auto& packet = image.as<hsa_kernel_dispatch_packet_t>();
      completion = &packet.completion_signal;
      entry.object = packet.kernel_object;
    } else if (type == HSA_PACKET_TYPE_VENDOR_SPECIFIC &&
               ((word >> 16U) & 0xffU) == ExtKernelDispatchFormat) {
      auto& packet = image.as<ExtDispatchPacket>();
      completion = &packet.completionSignal;
      entry.object = packet.kernelObject;
    } else if (type == HSA_PACKET_TYPE_VENDOR_SPECIFIC &&
               ((word >> 16U) & 0xffU) == HSA_AMD_PACKET_TYPE_BARRIER_VALUE) {
      auto& packet = image.as<hsa_amd_barrier_value_packet_t>();
      completion = &packet.completion_signal;
      entry.kernel = false;
      if (packet.cond == HSA_SIGNAL_CONDITION_LT) {
        entry.waitsFor = runtime_.queueOfCounter(packet.signal.handle);
      }
    }
    if (completion == nullptr || completion->handle != 0) {
      return;
    }
    entry.signal = runtime_.traceSignal();
    // (pushImage() counts it next)
    entry.index = created_ + 1;
    *completion = entry.signal;
    const std::lock_guard lock(traceMutex_);
    traced_.push_back(entry);
  }

  /**
   * With Runtime::extDispatch: a kernel goes as an extended dispatch packet, in clusters of one
   * workgroup, with all group memory (as CLR writes kernels on gfx12.5); with the barrier bit, it
   * takes the pending wait (see waitFor()) as its dependent signal. A kernel without the barrier
   * bit, and any other packet, gets the pending wait as a barrier-AND packet before it.
   */
  void extend(PacketImage& image) {
    std::uint32_t word = 0;
    std::memcpy(&word, image.bytes.data(), sizeof(word));
    const auto header = static_cast<std::uint16_t>(word & 0xffffU);
    constexpr auto TypeMask = static_cast<std::uint16_t>(((1U << HSA_PACKET_HEADER_WIDTH_TYPE) - 1U)
                                                         << HSA_PACKET_HEADER_TYPE);
    if (((header & TypeMask) >> HSA_PACKET_HEADER_TYPE) != HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      writePendingWait();
      return;
    }
    const auto standard = image.as<hsa_kernel_dispatch_packet_t>();
    // (the grid in workgroups, which the extended packet takes with 16 bits in y and z)
    const std::uint32_t groupsX = standard.grid_size_x / standard.workgroup_size_x;
    const std::uint32_t groupsY = standard.grid_size_y / standard.workgroup_size_y;
    const std::uint32_t groupsZ = standard.grid_size_z / standard.workgroup_size_z;
    if (groupsY > 0xffffU || groupsZ > 0xffffU) {
      warnOnce("kernels with more than 65535 workgroups in y or z go as kernel dispatch packets");
      writePendingWait();
      return;
    }
    ExtDispatchPacket ext{};
    ext.workgroupSizeX = standard.workgroup_size_x;
    ext.workgroupSizeY = standard.workgroup_size_y;
    ext.workgroupSizeZ = standard.workgroup_size_z;
    ext.clusterCountX = groupsX;
    ext.clusterCountY = static_cast<std::uint16_t>(groupsY);
    ext.clusterCountZ = static_cast<std::uint16_t>(groupsZ);
    ext.clusterSizeX = 1;
    ext.clusterSizeY = 1;
    ext.clusterSizeZ = 1;
    ext.perfHint = 127;
    ext.privateSegmentSize = standard.private_segment_size;
    ext.groupSegmentSize = standard.group_segment_size;
    ext.kernelObject = standard.kernel_object;
    ext.kernargAddress = standard.kernarg_address;
    ext.completionSignal = standard.completion_signal;
    auto extHeader = static_cast<std::uint16_t>(
        (header & ~TypeMask) | (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE));
    if (pendingWait_.handle != 0) {
      if ((header & BarrierBit) != 0) {
        ext.depSignal = pendingWait_;
        extHeader = withAcquire(extHeader, runtime_.waitFences);
        pendingWait_ = hsa_signal_t{0};
      } else {
        writePendingWait();
      }
    }
    // (the dimensions)
    const auto setup = static_cast<std::uint16_t>((word >> 16U) & 0xffU);
    image.as<ExtDispatchPacket>() = ext;
    image.setHeader(extHeader, static_cast<std::uint16_t>(ExtKernelDispatchFormat | (setup << 8U)));
  }

  /// the pending wait (see waitFor()) as a barrier-AND packet of its own
  void writePendingWait() {
    if (pendingWait_.handle == 0) {
      return;
    }
    PacketImage packet;
    packet.as<hsa_barrier_and_packet_t>().dep_signal[0] = pendingWait_;
    packet.setHeader(
        static_cast<std::uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                   BarrierBit | runtime_.waitFences),
        0);
    pendingWait_ = hsa_signal_t{0};
    pushImage(packet);
  }

  /// `header` with at least the acquire scope of `fences`
  static std::uint16_t withAcquire(std::uint16_t header, std::uint16_t fences) {
    constexpr unsigned Shift = HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    constexpr unsigned Mask = ((1U << HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE) - 1U) << Shift;
    const unsigned own = (header & Mask) >> Shift;
    const unsigned wanted = (fences & Mask) >> Shift;
    return static_cast<std::uint16_t>((header & ~Mask) | (std::max(own, wanted) << Shift));
  }

  public:
  void pushImage(const PacketImage& packet) {
    ++created_;
    if (!overflow_.empty()) {
      drain();
    }
    if (overflow_.empty() && space()) {
      write(packet);
      return;
    }
    if (overflow_.empty()) {
      warnOnce("an HSA queue is full, its packets wait in host memory (DEVICE_HSA_QUEUE_PACKETS)");
      runtime_.hostThread.drainUntilEmpty(this, [this]() {
        const std::lock_guard lock(mutex);
        return drain();
      });
    }
    overflow_.push_back(packet);
  }

  /// writes the packets that wait in host memory into the queue, as far as there is space; returns
  /// whether some still wait
  bool drain() {
    while (!overflow_.empty() && space()) {
      write(overflow_.front());
      overflow_.pop_front();
    }
    return !overflow_.empty();
  }

  /// memory for the arguments of a kernel, from the ring; valid until the next mark is reached
  void* kernargs(std::size_t size, std::size_t alignment) {
    size = std::max<std::size_t>(size, 1);
    for (int attempt = 0; attempt < 2; ++attempt) {
      auto start = alignUp(head_, std::max<std::size_t>(alignment, 16));
      if (start / ringBytes_ != (start + size - 1) / ringBytes_) {
        // an allocation does not wrap around
        start = alignUp(start, ringBytes_);
      }
      if (start + size - tail_ <= ringBytes_) {
        head_ = start + size;
        return ring_ + (start % ringBytes_);
      }
      reclaim();
    }
    warnOnce("the memory for kernel arguments of an HSA queue is used up; further arguments get "
             "memory of their own (DEVICE_HSA_QUEUE_KERNARG_MIB)");
    void* memory = allocKernargMemory(size);
    ownKernargs_.emplace_back(memory, size);
    return memory;
  }

  /**
   * Enqueues a mark: a barrier packet that completes once all work before it on the queue has, and
   * then makes it visible to the host and other agents. Returns its number.
   */
  std::uint64_t mark() {
    std::uint16_t counterFences = runtime_.markFences;
    if (runtime_.markSignals) {
      // first the signal of the mark, for the queues that wait for it (see waitFor()); then the
      // counter, which needs no fences of its own after that
      const auto signal = takeMarkSignal(marks_ + 1);
      PacketImage own;
      own.as<hsa_barrier_and_packet_t>().completion_signal = signal;
      own.setHeader(
          static_cast<std::uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                     BarrierBit | runtime_.markFences),
          0);
      push(own);
      counterFences = 0;
    }
    PacketImage packet;
    packet.as<hsa_barrier_and_packet_t>().completion_signal = counter_;
    packet.setHeader(
        static_cast<std::uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                   BarrierBit | counterFences),
        0);
    push(packet);
    ++marks_;
    lastMarkIndex_ = created_;
    if (runtime_.schedule) {
      // (see predictedFreeLocked(); a queue that nobody asks keeps the latest ones)
      markPredictions_.emplace_back(marks_, predictedFree_.load(std::memory_order_relaxed));
      if (markPredictions_.size() > MaxMarkPredictions) {
        markPredictions_.pop_front();
      }
    }
    releases_.emplace_back(marks_, head_);
    if (!ownKernargs_.empty()) {
      ownReleases_.emplace_back(marks_, std::move(ownKernargs_));
      ownKernargs_.clear();
    }
    return marks_;
  }

  /// a mark after the packets before `index` (see created()): the latest one if it is, or a new one
  std::uint64_t markCoveringLocked(std::uint64_t index) {
    if (pendingWait_.handle != 0) {
      // (a wait not written yet counts as work after the latest mark)
      return mark();
    }
    if (index == 0) {
      return 0;
    }
    if (marks_ > 0 && lastMarkIndex_ >= index) {
      return marks_;
    }
    return mark();
  }

  /// whether the queue has done all it got (its last packet is a mark that has been reached), with
  /// `mutex` held
  [[nodiscard]] bool idleLocked() const {
    return pendingWait_.handle == 0 && overflow_.empty() && marks_ > 0 &&
           lastMarkIndex_ == created_ && reached(marks_);
  }

  // the time (s of steady_clock) by which the queue is predicted to have done all it got, for
  // DEVICE_HSA_QUEUE_SCHEDULE (see Program::submitScheduled()): its work as learned or guessed, and
  // the predicted work of the queues it waits for (all of it: a bound)

  /// with `mutex` held
  [[nodiscard]] double predictedFreeLocked(double now) {
    if (idleLocked()) {
      predictedFree_.store(now, std::memory_order_relaxed);
      markPredictions_.clear();
      return now;
    }
    // anchored to what the queue has done: by how much the latest mark it has reached was predicted
    // later than now (it is ahead), or the next one earlier than now (it is behind); a busy queue
    // that never gets idle would drift away otherwise (all the work of the queues it waits for)
    const auto reachedMarks =
        static_cast<std::uint64_t>(CounterStart - hsa_signal_load_relaxed(counter_));
    double shift = 0;
    bool passed = false;
    double latest = 0;
    while (!markPredictions_.empty() && markPredictions_.front().first <= reachedMarks) {
      latest = markPredictions_.front().second;
      passed = true;
      markPredictions_.pop_front();
    }
    if (passed && latest > now) {
      shift = now - latest;
    } else if (!markPredictions_.empty() && markPredictions_.front().second < now) {
      shift = now - markPredictions_.front().second;
    }
    if (shift != 0) {
      predictedFree_.store(predictedFree_.load(std::memory_order_relaxed) + shift,
                           std::memory_order_relaxed);
      for (auto& prediction : markPredictions_) {
        prediction.second += shift;
      }
    }
    return std::max(predictedFree_.load(std::memory_order_relaxed), now);
  }

  [[nodiscard]] double predictedFree() const {
    return predictedFree_.load(std::memory_order_relaxed);
  }

  /// with `mutex` held
  void predictWork(double now, double seconds) {
    predictedFree_.store(std::max(predictedFree_.load(std::memory_order_relaxed), now) + seconds,
                         std::memory_order_relaxed);
  }

  /// with `mutex` held
  void predictUntil(double time) {
    if (time > predictedFree_.load(std::memory_order_relaxed)) {
      predictedFree_.store(time, std::memory_order_relaxed);
    }
  }

  [[nodiscard]] bool reached(std::uint64_t mark) const {
    return mark == 0 ||
           hsa_signal_load_scacquire(counter_) <= CounterStart - static_cast<std::int64_t>(mark);
  }

  /// (without `mutex` held: the host thread may have to write the packets up to the mark first)
  void wait(std::uint64_t mark) const {
    if (mark == 0) {
      return;
    }
    const auto below = CounterStart - static_cast<std::int64_t>(mark) + 1;
    while (hsa_signal_wait_scacquire(
               counter_, HSA_SIGNAL_CONDITION_LT, below, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) >=
           below) {
    }
  }

  /**
   * The signal of mark `mark` (Runtime::markSignals), for a packet of `waiter` that is about to
   * wait for it, before mark `after` of the waiter: the signal stays the mark's until the waiter
   * has reached that mark. Nothing (a handle of 0) if the mark has been reached and its signal
   * given back. (Locks only the signals of the marks, not `mutex`.)
   */
  hsa_signal_t markSignalFor(std::uint64_t mark, const Queue& waiter, std::uint64_t after) {
    const std::lock_guard lock(signalMutex_);
    for (auto& entry : markSignals_) {
      if (entry.mark == mark) {
        entry.waiters.emplace_back(&waiter, after);
        return entry.signal;
      }
      if (entry.mark > mark) {
        break;
      }
    }
    return hsa_signal_t{0};
  }

  /// lets the work enqueued from now on wait until the value of the signal, masked by `mask` (e.g.
  /// 0xffffffff for a 32-bit counter), reaches `value`
  void waitForCounter(hsa_signal_t signal, std::int64_t value, std::int64_t mask) {
    PacketImage packet;
    auto& barrier = packet.as<hsa_amd_barrier_value_packet_t>();
    barrier.signal = signal;
    barrier.value = value;
    barrier.mask = mask;
    barrier.cond = HSA_SIGNAL_CONDITION_GTE;
    // (the host, or a stream of HIP, has written the data that the counter stands for)
    packet.setHeader(
        static_cast<std::uint16_t>((HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) |
                                   BarrierBit | hostWaitFences()),
        HSA_AMD_PACKET_TYPE_BARRIER_VALUE);
    push(packet);
    acquireFromHost();
  }

  /// the fences of a packet that waits for the host (see Runtime::acquireByKernel)
  [[nodiscard]] std::uint16_t hostWaitFences() const {
    return runtime_.acquireByKernel ? QueueWaitFences : WaitFences;
  }

  /// after a packet that waited for the host: lets the work after it see what the host wrote
  void acquireFromHost() {
    if (runtime_.acquireByKernel) {
      dispatch(kernelOf(reinterpret_cast<const void*>(hsaAcquireSystem)),
               dim3(1),
               dim3(1),
               0,
               nullptr,
               KernelFences,
               hsa_signal_t{0});
    }
  }

  /// lets the work enqueued from now on wait until mark `mark` of `other` has been reached
  void waitFor(Queue& other, std::uint64_t mark) {
    if (&other == this || mark == 0 || other.reached(mark)) {
      return;
    }
    if (runtime_.schedule) {
      predictUntil(other.predictedFree());
    }
    if (runtime_.markSignals) {
      // (the packet comes next, before the next mark of this queue: registered before the packet
      // gets written, the signal stays the mark's until this queue has reached that mark)
      const auto signal = other.markSignalFor(mark, *this, marks_ + 1);
      if (signal.handle == 0) {
        // reached, and its signal given back meanwhile
        return;
      }
      if (runtime_.extDispatch) {
        // the next kernel with the barrier bit takes it as its dependent signal (see extend()); a
        // wait pending already goes as a packet of its own, as a kernel takes one signal only
        writePendingWait();
        pendingWait_ = signal;
        return;
      }
      PacketImage packet;
      packet.as<hsa_barrier_and_packet_t>().dep_signal[0] = signal;
      packet.setHeader(
          static_cast<std::uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                     BarrierBit | runtime_.waitFences),
          0);
      push(packet);
      return;
    }
    const auto below = CounterStart - static_cast<std::int64_t>(mark) + 1;
    if (runtime_.barrierValue) {
      PacketImage packet;
      auto& barrier = packet.as<hsa_amd_barrier_value_packet_t>();
      barrier.signal = other.counter_;
      barrier.value = below;
      barrier.mask = -1;
      barrier.cond = HSA_SIGNAL_CONDITION_LT;
      packet.setHeader(
          static_cast<std::uint16_t>((HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) |
                                     BarrierBit | runtime_.waitFences),
          HSA_AMD_PACKET_TYPE_BARRIER_VALUE);
      push(packet);
    } else {
      // the command processor does not wait for values: a kernel does
      const auto* value = const_cast<const std::int64_t*>(other.counterValue_);
      void* args[] = {&value, const_cast<std::int64_t*>(&below)};
      dispatch(kernelOf(reinterpret_cast<const void*>(hsaWaitBelow)),
               dim3(1),
               dim3(1),
               0,
               args,
               runtime_.waitFences,
               hsa_signal_t{0});
    }
  }

  static const KernelInfo& kernelOf(const void* function) {
    return Runtime::instance().kernel(function);
  }

  void fillKernargs(const KernelInfo& kernel,
                    void* kernargs,
                    dim3 grid,
                    dim3 block,
                    std::size_t sharedMemBytes,
                    void** args) const {
    auto* bytes = static_cast<std::uint8_t*>(kernargs);
    std::memset(bytes, 0, kernel.kernargSize);
    const auto* amdQueue = reinterpret_cast<const amd_queue_t*>(queue_);
    std::size_t explicitIndex = 0;
    for (const auto& arg : kernel.args) {
      const auto put = [&](std::uint64_t value) {
        std::memcpy(bytes + arg.offset, &value, std::min<std::size_t>(arg.size, sizeof(value)));
      };
      switch (arg.kind) {
      case ArgKind::Explicit:
        std::memcpy(bytes + arg.offset, args[explicitIndex], arg.size);
        ++explicitIndex;
        break;
      case ArgKind::BlockCountX:
        put(grid.x);
        break;
      case ArgKind::BlockCountY:
        put(grid.y);
        break;
      case ArgKind::BlockCountZ:
        put(grid.z);
        break;
      case ArgKind::GroupSizeX:
        put(block.x);
        break;
      case ArgKind::GroupSizeY:
        put(block.y);
        break;
      case ArgKind::GroupSizeZ:
        put(block.z);
        break;
      case ArgKind::GridDims:
        // HIP launches in three dimensions
        put(3);
        break;
      case ArgKind::DynamicLdsSize:
        put(sharedMemBytes);
        break;
      case ArgKind::QueuePtr:
        put(reinterpret_cast<std::uint64_t>(queue_));
        break;
      case ArgKind::PrivateBase:
        put(amdQueue->private_segment_aperture_base_hi);
        break;
      case ArgKind::SharedBase:
        put(amdQueue->group_segment_aperture_base_hi);
        break;
      case ArgKind::Zero:
      case ArgKind::Unsupported:
      case ArgKind::Unknown:
        break;
      }
    }
  }

  /// the dispatch packet of a kernel, without its header
  [[nodiscard]] hsa_kernel_dispatch_packet_t dispatchPacket(const KernelInfo& kernel,
                                                            dim3 grid,
                                                            dim3 block,
                                                            std::size_t sharedMemBytes,
                                                            void* kernargs,
                                                            hsa_signal_t completion) const {
    hsa_kernel_dispatch_packet_t packet{};
    packet.workgroup_size_x = static_cast<std::uint16_t>(block.x);
    packet.workgroup_size_y = static_cast<std::uint16_t>(block.y);
    packet.workgroup_size_z = static_cast<std::uint16_t>(block.z);
    packet.grid_size_x = grid.x * block.x;
    packet.grid_size_y = grid.y * block.y;
    packet.grid_size_z = grid.z * block.z;
    packet.private_segment_size =
        kernel.dynamicStack ? std::max(kernel.privateSize, runtime_.stackSize) : kernel.privateSize;
    packet.group_segment_size = kernel.groupSize + static_cast<std::uint32_t>(sharedMemBytes);
    packet.kernel_object = kernel.object;
    packet.kernarg_address = kernargs;
    packet.completion_signal = completion;
    return packet;
  }

  /// enqueues a kernel; it waits for all work before it on the queue
  void dispatch(const KernelInfo& kernel,
                dim3 grid,
                dim3 block,
                std::size_t sharedMemBytes,
                void** args,
                std::uint16_t fences,
                hsa_signal_t completion) {
    if (kernel.unsupported) {
      warnOnce("the kernel " + kernel.name +
               " may use printf, hostcalls, the device heap or device enqueue, which HSA queues do "
               "not provide");
    }
    // (HIP refuses such a launch as well)
    const auto workItems = static_cast<std::uint64_t>(block.x) * block.y * block.z;
    if (kernel.maxWorkgroupSize > 0 && workItems > kernel.maxWorkgroupSize) {
      logError() << "HSA queues: the kernel" << kernel.name << "got launched with" << workItems
                 << "work-items per workgroup, more than the" << kernel.maxWorkgroupSize
                 << "it was compiled for (__launch_bounds__).";
    }
    void* kernargs = this->kernargs(kernel.kernargSize, kernel.kernargAlign);
    fillKernargs(kernel, kernargs, grid, block, sharedMemBytes, args);
    flushKernargs(static_cast<const char*>(kernargs) +
                  std::max<std::uint32_t>(kernel.kernargSize, 1) - 1);
    PacketImage packet;
    packet.as<hsa_kernel_dispatch_packet_t>() =
        dispatchPacket(kernel, grid, block, sharedMemBytes, kernargs, completion);
    packet.setHeader(
        static_cast<std::uint16_t>((HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                                   BarrierBit | fences),
        static_cast<std::uint16_t>(3U << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS));
    push(packet);
  }

  /// enqueues a host function: the queue waits while it runs (see HostThread)
  void hostFunction(const std::function<void()>& function) {
    auto [reach, gate] = runtime_.hostThread.signals();
    PacketImage barrier;
    barrier.as<hsa_barrier_and_packet_t>().completion_signal = reach;
    barrier.setHeader(
        static_cast<std::uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                   BarrierBit | MarkFences),
        0);
    push(barrier);
    if (runtime_.barrierValue) {
      PacketImage packet;
      auto& wait = packet.as<hsa_amd_barrier_value_packet_t>();
      wait.signal = gate;
      wait.value = 1;
      wait.mask = -1;
      wait.cond = HSA_SIGNAL_CONDITION_GTE;
      wait.completion_signal = reach;
      packet.setHeader(
          static_cast<std::uint16_t>((HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) |
                                     BarrierBit | hostWaitFences()),
          HSA_AMD_PACKET_TYPE_BARRIER_VALUE);
      push(packet);
    } else if (runtime_.markSignals) {
      // the gate drops to 0 there (see HostThread::gateDrops)
      PacketImage packet;
      auto& wait = packet.as<hsa_barrier_and_packet_t>();
      wait.dep_signal[0] = gate;
      wait.completion_signal = reach;
      packet.setHeader(
          static_cast<std::uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                     BarrierBit | hostWaitFences()),
          0);
      push(packet);
    } else {
      volatile hsa_signal_value_t* gateValue = nullptr;
      check(hsa_amd_signal_value_pointer(gate, &gateValue), "hsa_amd_signal_value_pointer");
      const auto* value = const_cast<const std::int64_t*>(gateValue);
      std::int64_t atLeast = 1;
      void* args[] = {&value, &atLeast};
      dispatch(kernelOf(reinterpret_cast<const void*>(hsaWaitAtLeast)),
               dim3(1),
               dim3(1),
               0,
               args,
               hostWaitFences(),
               reach);
    }
    acquireFromHost();
    runtime_.hostThread.push({reach, gate, function});
  }

  /// enqueues a kernel that takes the time after all work before it, into the signal
  void stamp(hsa_signal_t signal) {
    if (!profiling_) {
      check(hsa_amd_profiling_set_profiler_enabled(queue_, 1),
            "hsa_amd_profiling_set_profiler_enabled");
      profiling_ = true;
    }
    hsa_signal_store_relaxed(signal, 1);
    dispatch(kernelOf(reinterpret_cast<const void*>(hsaStamp)),
             dim3(1),
             dim3(1),
             0,
             nullptr,
             KernelFences,
             signal);
  }

  /// memory of at least `bytes`, for the temporaries of the graphs that run on the queue
  void* arena(std::size_t bytes) {
    if (bytes > arenaBytes_) {
      // the graphs translated before keep the arena they got
      void* arena = nullptr;
      APIWRAP(hipMalloc(&arena, bytes));
      arenas_.emplace_back(arena, bytes);
      runtime_.deviceBytes += bytes;
      arenaBytes_ = bytes;
    }
    return arenas_.empty() ? nullptr : arenas_.back().first;
  }

  void* allocAsync(std::size_t bytes) {
    void* ptr = nullptr;
    APIWRAP(hipMallocAsync(&ptr, bytes, helper_));
    return ptr;
  }

  /// frees once the queue has passed the packets before `index`
  void freeAsync(void* ptr, std::uint64_t index) {
    pendingFrees_.emplace_back(markCoveringLocked(index), ptr);
    reclaim();
  }

  /// releases the arguments and frees the allocations that the marks reached so far allow
  void reclaim() {
    while (!releases_.empty() && reached(releases_.front().first)) {
      tail_ = releases_.front().second;
      releases_.pop_front();
    }
    while (!ownReleases_.empty() && reached(ownReleases_.front().first)) {
      for (const auto& [memory, bytes] : ownReleases_.front().second) {
        freeKernargMemory(memory, bytes);
      }
      ownReleases_.pop_front();
    }
    while (!pendingFrees_.empty() && reached(pendingFrees_.front().first)) {
      APIWRAP(hipFreeAsync(pendingFrees_.front().second, helper_));
      pendingFrees_.pop_front();
    }
  }

  /// the number of packets enqueued so far (also those that wait in host memory)
  [[nodiscard]] std::uint64_t created() const { return created_; }

  std::mutex mutex;

  private:
  /// a signal for mark `mark`, after giving back those that no packet reads any longer
  hsa_signal_t takeMarkSignal(std::uint64_t mark) {
    const std::lock_guard lock(signalMutex_);
    // (the marks reached so far: the counter comes after their signals; nothing goes back while
    // the queues go, whose counters tell about the waiters)
    const auto reachedMarks =
        runtime_.finalizing.load(std::memory_order_relaxed)
            ? std::uint64_t{0}
            : static_cast<std::uint64_t>(CounterStart - hsa_signal_load_scacquire(counter_));
    for (auto entry = markSignals_.begin();
         entry != markSignals_.end() && entry->mark <= reachedMarks;) {
      auto& waiters = entry->waiters;
      waiters.erase(
          std::remove_if(waiters.begin(),
                         waiters.end(),
                         [](const auto& waiter) { return waiter.first->reached(waiter.second); }),
          waiters.end());
      if (waiters.empty()) {
        runtime_.freeMarkSignal(entry->signal);
        entry = markSignals_.erase(entry);
      } else {
        ++entry;
      }
    }
    const auto signal = runtime_.markSignal();
    markSignals_.push_back(MarkSignal{mark, signal, {}});
    return signal;
  }

  [[nodiscard]] bool space() const {
    return hsa_queue_load_write_index_relaxed(queue_) + 1 -
               hsa_queue_load_read_index_scacquire(queue_) <=
           queue_->size;
  }

  void write(const PacketImage& packet) {
    const auto index = hsa_queue_add_write_index_relaxed(queue_, 1);
    auto* slot = static_cast<std::uint8_t*>(queue_->base_address) + (index % queue_->size) * 64;
    // all but the header, which makes the packet valid
    std::memcpy(slot + sizeof(std::uint32_t),
                packet.bytes.data() + sizeof(std::uint32_t),
                packet.bytes.size() - sizeof(std::uint32_t));
    std::uint32_t word = 0;
    std::memcpy(&word, packet.bytes.data(), sizeof(word));
    __atomic_store_n(reinterpret_cast<std::uint32_t*>(slot), word, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(queue_->doorbell_signal, static_cast<hsa_signal_value_t>(index));
  }

  Runtime& runtime_;
  int number_;
  hsa_queue_t* queue_{nullptr};
  hsa_signal_t counter_{};
  // the signals of the marks not given back yet (Runtime::markSignals), in the order of the marks,
  // and the queues whose packets wait for each, with the mark of theirs after the packet
  struct MarkSignal {
    std::uint64_t mark{0};
    hsa_signal_t signal{};
    std::vector<std::pair<const Queue*, std::uint64_t>> waiters;
  };
  std::mutex signalMutex_;
  std::deque<MarkSignal> markSignals_;
  // with Runtime::extDispatch: the signal of a mark that the work enqueued next waits for, not
  // written yet (see extend())
  hsa_signal_t pendingWait_{};
  volatile hsa_signal_value_t* counterValue_{nullptr};
  std::uint64_t created_{0};
  std::uint64_t marks_{0};
  std::uint64_t lastMarkIndex_{0};
  std::atomic<double> predictedFree_{0.0};
  // the marks not reached yet when last asked, and the predicted time of each
  std::deque<std::pair<std::uint64_t, double>> markPredictions_;
  std::deque<PacketImage> overflow_;
  bool profiling_{false};
  std::mutex traceMutex_;
  std::deque<TraceEntry> traced_;

  char* ring_{nullptr};
  std::uint64_t ringBytes_{0};
  std::uint64_t head_{0};
  std::uint64_t tail_{0};
  // the marks, and how far the ring had been handed out when each got enqueued
  std::deque<std::pair<std::uint64_t, std::uint64_t>> releases_;
  // arguments that did not fit into the ring, and the marks that free them
  std::vector<std::pair<void*, std::size_t>> ownKernargs_;
  std::deque<std::pair<std::uint64_t, std::vector<std::pair<void*, std::size_t>>>> ownReleases_;

  // for asynchronous allocations, which a stream of HIP that never gets any work makes
  hipStream_t helper_{nullptr};
  std::deque<std::pair<std::uint64_t, void*>> pendingFrees_;
  std::vector<std::pair<void*, std::size_t>> arenas_;
  std::size_t arenaBytes_{0};
};

} // namespace device::hsaqueues

#endif // SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_QUEUE_H_
