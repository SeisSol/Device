// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HIP_OWNQUEUES_HOSTTHREAD_H_
#define SEISSOLDEVICE_INTERFACES_HIP_OWNQUEUES_HOSTTHREAD_H_

#include "Common.h"

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

namespace device::ownqueues {

/**
 * Runs host functions for the queues: each once its queue has reached it (`reach` drops to 0), in
 * the order the queues get there; then it lets the queue go on (`gate` rises to 1). The queue drops
 * `reach` once more when it has gone on, after which both signals get reused.
 */
class HostThread {
  public:
  struct Job {
    hsa_signal_t reach;
    hsa_signal_t gate;
    std::function<void()> function;
    bool ran{false};
  };

  /**
   * Lets the thread write the packets of a queue that wait in host memory into it, as space gets
   * free: `drain` does that, and returns whether some still wait.
   */
  void drainUntilEmpty(const void* queue, std::function<bool()> drain) {
    {
      const std::lock_guard lock(mutex_);
      if (!thread_.joinable()) {
        thread_ = std::thread([this]() { run(); });
      }
      drains_.emplace(queue, std::move(drain));
    }
    condition_.notify_one();
  }

  ~HostThread() { stop(); }

  std::pair<hsa_signal_t, hsa_signal_t> signals() {
    const std::lock_guard lock(mutex_);
    if (!free_.empty()) {
      auto pair = free_.back();
      free_.pop_back();
      return pair;
    }
    // (the thread polls them, and a kernel that waits for the gate reads its value)
    hsa_signal_t reach;
    hsa_signal_t gate;
    check(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &reach),
          "hsa_amd_signal_create");
    check(hsa_amd_signal_create(0, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &gate),
          "hsa_amd_signal_create");
    return {reach, gate};
  }

  void push(Job job) {
    {
      const std::lock_guard lock(mutex_);
      if (!thread_.joinable()) {
        thread_ = std::thread([this]() { run(); });
      }
      jobs_.push_back(std::move(job));
    }
    condition_.notify_one();
  }

  void stop() {
    {
      const std::lock_guard lock(mutex_);
      stop_ = true;
    }
    condition_.notify_one();
    if (thread_.joinable()) {
      thread_.join();
    }
    for (auto& [reach, gate] : free_) {
      hsa_signal_destroy(reach);
      hsa_signal_destroy(gate);
    }
    free_.clear();
  }

  private:
  void run() {
    // the jobs whose queues have not reached them yet, or have not gone on after them yet; and the
    // queues with packets that wait in host memory
    std::vector<Job> active;
    std::map<const void*, std::function<bool()>> drains;
    while (true) {
      {
        std::unique_lock lock(mutex_);
        // (the queues whose packets still wait count as well: they get space as the device goes on)
        if (active.empty() && drains.empty() && drains_.empty()) {
          condition_.wait(lock, [this]() { return stop_ || !jobs_.empty() || !drains_.empty(); });
        }
        if (stop_ && jobs_.empty() && active.empty() && drains.empty() && drains_.empty()) {
          return;
        }
        while (!jobs_.empty()) {
          active.push_back(std::move(jobs_.front()));
          jobs_.pop_front();
        }
        drains.insert(drains_.begin(), drains_.end());
        drains_.clear();
      }
      bool progress = false;
      for (auto drain = drains.begin(); drain != drains.end();) {
        if (!drain->second()) {
          drain = drains.erase(drain);
          progress = true;
        } else {
          ++drain;
        }
      }
      for (auto job = active.begin(); job != active.end();) {
        if (!job->ran && hsa_signal_load_scacquire(job->reach) < 1) {
          job->function();
          hsa_signal_store_screlease(job->gate, 1);
          job->ran = true;
          progress = true;
        }
        if (job->ran && hsa_signal_load_scacquire(job->reach) < 0) {
          hsa_signal_store_relaxed(job->reach, 1);
          hsa_signal_store_relaxed(job->gate, 0);
          {
            const std::lock_guard lock(mutex_);
            free_.emplace_back(job->reach, job->gate);
          }
          job = active.erase(job);
          progress = true;
          continue;
        }
        ++job;
      }
      if (!progress) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Job> jobs_;
  std::map<const void*, std::function<bool()>> drains_;
  std::vector<std::pair<hsa_signal_t, hsa_signal_t>> free_;
  std::thread thread_;
  bool stop_{false};
};

} // namespace device::ownqueues

#endif // SEISSOLDEVICE_INTERFACES_HIP_OWNQUEUES_HOSTTHREAD_H_
