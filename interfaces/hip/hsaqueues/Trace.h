// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_TRACE_H_
#define SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_TRACE_H_

#include "Common.h"
#include "HsaQueues.h"
#include "Queue.h"
#include "Runtime.h"

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
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace device::hsaqueues {

/**
 * DEVICE_HSA_QUEUE_DAG_LOG=<prefix>: while the kernels get traced (see trace()), what the streams
 * do and when the kernels ran, per process in <prefix>-<rank>.txt (the rank from SLURM_PROCID,
 * PMI_RANK or OMPI_COMM_WORLD_RANK, else the process id), as the work and its dependencies for a
 * schedule over the queues. Times in ns of steady_clock, the packets of a queue counted from 1; a
 * line for each of:
 * - L t stream queue first last name: a kernel launched on a stream (packets first+1..last)
 * - G t stream queue first last program: a graph launched on a stream (see P)
 * - P program queue levels...: the level of each kernel of the graph (h: a host function)
 * - R t stream queue event mark index: an event recorded on a stream (its mark, after packet index)
 * - W t stream queue event: a stream waits for an event
 * - O t stream queue first last kind: other work on a stream (copy, host, counter)
 * - K queue index start end name: the times a kernel ran
 * - A queue index start end target: the times of a wait (target: a queue, or -1 for the host)
 * - N stream name: the name of a stream (see nameStream())
 * - S t program queue level start end alone: a level that a graph spreads over queues
 *   (DEVICE_HSA_QUEUE_SCHEDULE), its predicted start and end, and its end on its queue alone
 * - F queue index program level free: a wait of a queue for the fork of such a level, and when
 *   the queue was predicted to be free
 * - J queue index program level: a wait of the queue of the level for the join of a helper
 */
class DagLog {
  public:
  [[nodiscard]] bool on() const {
    return !prefix_.empty() && Runtime::instance().tracing.load(std::memory_order_relaxed);
  }

  template <typename... Args>
  void line(const char* format, Args... args) {
    const std::lock_guard lock(mutex_);
    if (file_ == nullptr) {
      std::string rank = std::to_string(static_cast<long>(getpid()));
      for (const char* variable : {"SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK"}) {
        if (const char* value = std::getenv(variable); value != nullptr) {
          rank = value;
          break;
        }
      }
      file_ = std::fopen((prefix_ + "-" + rank + ".txt").c_str(), "w");
      if (file_ == nullptr) {
        prefix_.clear();
        return;
      }
      std::setvbuf(file_, nullptr, _IOFBF, 1 << 20);
    }
    std::fprintf(file_, format, args...);
  }

  void flush() {
    const std::lock_guard lock(mutex_);
    if (file_ != nullptr) {
      std::fflush(file_);
    }
  }

  /// whether the program still has to be described (once)
  bool firstTime(const void* program) {
    const std::lock_guard lock(mutex_);
    return described_.insert(program).second;
  }

  static long long now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  static long long ns(std::chrono::steady_clock::time_point time) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
  }

  private:
  std::string prefix_{setting<std::string>("DAG_LOG", "")};
  std::mutex mutex_;
  std::FILE* file_{nullptr};
  std::unordered_set<const void*> described_;
};

DagLog& dagLog();

/**
 * Takes the times of the traced packets once they have completed, and hands them to the sink on
 * the clock of the host: the system timestamps of HSA, mapped onto std::chrono::steady_clock by
 * reading both together again and again.
 */
class Tracer {
  public:
  void start(TraceSink sink) {
    stop();
    auto& runtime = Runtime::instance();
    sink_ = std::move(sink);
    for (auto* queue : runtime.allQueues()) {
      const std::lock_guard lock(queue->mutex);
      queue->enableProfiling();
    }
    stop_.store(false);
    runtime.tracing.store(true);
    thread_ = std::thread([this]() { run(); });
  }

  /// once all packets traced so far have been reported (or after ten seconds)
  void stop() {
    if (!thread_.joinable()) {
      return;
    }
    Runtime::instance().tracing.store(false);
    stop_.store(true);
    thread_.join();
    sink_ = nullptr;
  }

  private:
  void calibrate() {
    auto best = std::chrono::steady_clock::duration::max();
    for (int attempt = 0; attempt < 3; ++attempt) {
      const auto before = std::chrono::steady_clock::now();
      std::uint64_t ticks = 0;
      hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &ticks);
      const auto after = std::chrono::steady_clock::now();
      if (after - before < best) {
        best = after - before;
        ticks_ = ticks;
        time_ = before + (after - before) / 2;
      }
    }
    calibrated_ = std::chrono::steady_clock::now();
  }

  [[nodiscard]] std::chrono::steady_clock::time_point host(std::uint64_t ticks) const {
    const auto seconds = (static_cast<double>(ticks) - static_cast<double>(ticks_)) /
                         static_cast<double>(Runtime::instance().timestampFrequency);
    return time_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                       std::chrono::duration<double>(seconds));
  }

  void run() {
    auto& runtime = Runtime::instance();
    calibrate();
    std::vector<Queue::TraceEntry> done;
    std::optional<std::chrono::steady_clock::time_point> stopping;
    while (true) {
      if (std::chrono::steady_clock::now() - calibrated_ > std::chrono::milliseconds(50)) {
        calibrate();
      }
      bool pending = false;
      bool reported = false;
      for (auto* queue : runtime.allQueues()) {
        done.clear();
        queue->takeTraced(done);
        pending = pending || !queue->traceEmpty();
        for (const auto& entry : done) {
          hsa_amd_profiling_dispatch_time_t time{};
          if (hsa_amd_profiling_get_dispatch_time(runtime.gpu, entry.signal, &time) ==
                  HSA_STATUS_SUCCESS &&
              time.start >= runtime.startTimestamp && time.end >= time.start) {
            const auto name = entry.kernel     ? runtime.kernelName(entry.object)
                              : entry.waitsFor ? "wait for queue " + std::to_string(*entry.waitsFor)
                                               : std::string("wait for the host");
            sink_(name, queue->number(), host(time.start), host(time.end));
            if (dagLog().on()) {
              if (entry.kernel) {
                dagLog().line("K %d %llu %lld %lld %s\n",
                              queue->number(),
                              static_cast<unsigned long long>(entry.index),
                              DagLog::ns(host(time.start)),
                              DagLog::ns(host(time.end)),
                              name.c_str());
              } else {
                dagLog().line("A %d %llu %lld %lld %d\n",
                              queue->number(),
                              static_cast<unsigned long long>(entry.index),
                              DagLog::ns(host(time.start)),
                              DagLog::ns(host(time.end)),
                              entry.waitsFor ? *entry.waitsFor : -1);
              }
            }
          }
          runtime.freeTraceSignal(entry.signal);
          reported = true;
        }
      }
      if (stop_.load()) {
        if (!stopping) {
          stopping = std::chrono::steady_clock::now();
        }
        if (!pending || std::chrono::steady_clock::now() - *stopping > std::chrono::seconds(10)) {
          return;
        }
      }
      if (!reported) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    }
  }

  TraceSink sink_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::uint64_t ticks_{0};
  std::chrono::steady_clock::time_point time_;
  std::chrono::steady_clock::time_point calibrated_;
};

Tracer& tracer();

} // namespace device::hsaqueues

#endif // SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_TRACE_H_
