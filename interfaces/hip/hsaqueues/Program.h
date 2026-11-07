// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_PROGRAM_H_
#define SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_PROGRAM_H_

#include "Common.h"
#include "HsaQueues.h"
#include "Queue.h"

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

struct ProgramPacket {
  hsa_kernel_dispatch_packet_t packet{};
  std::uint16_t header{0};
  std::uint16_t rest{0};
  // a host function instead of a kernel
  std::function<void()> host;
  // the level in the graph
  std::size_t level{0};
};

/// the HSA queue number of helper lane `lane` of the programs of queue `queue`
inline int helperQueueNumber(int queue, unsigned lane) {
  return FirstHelperQueue + (queue + 1000) * 16 + static_cast<int>(lane);
}

/**
 * A graph translated for a queue. The kernels of a level of the graph are independent of each
 * other and go without barriers between them; a queue runs them one after the other though (MI250X,
 * ROCm 6.3).
 *
 * Experimental, and off by default (they make some setups faster and others slower):
 * - DEVICE_HSA_QUEUE_SCHEDULE=1: the levels get spread over other HSA queues by a predicted
 *   schedule (see submitScheduled()).
 * - DEVICE_HSA_QUEUE_GRAPH_ORIGIN=1: each kernel goes back onto the queue of the stream whose work
 *   it was (lanes); the program takes the mutexes of those queues while holding that of its own,
 *   so graphs that get launched from several threads at once may wait for each other forever.
 * - DEVICE_HSA_QUEUE_GRAPH_SPREAD=k: the program gets spread over its queue and k - 1 helper queues
 *   of its own by the durations that its first replays measure (see buildSpread()).
 */
class Program {
  public:
  /// `origins`: the HSA queue of the stream whose work each node is, where known (see noteOrigin())
  Program(Queue& queue, hipGraph_t graph, const std::unordered_map<hipGraphNode_t, int>& origins);
  ~Program();

  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  /// enqueues the program on its queue, with the queue's mutex held; returns the write index after
  std::uint64_t submit();

  /// the level of each of its kernels (h: a host function), for the DAG log
  [[nodiscard]] std::string levels() const {
    std::string text;
    for (const auto& packet : packets_) {
      text += packet.host ? " h" : " " + std::to_string(packet.level);
    }
    return text;
  }

  private:
  /// the lanes `lanes` wait until the signal has counted down `count` times
  void waitOnLanes(hsa_signal_t signal,
                   std::int64_t count,
                   const std::vector<unsigned>& lanes,
                   bool byKernel = false);
  void push(unsigned lane, const PacketImage& image);

  Queue& queue_;
  std::vector<ProgramPacket> packets_;
  void* kernargs_{nullptr};
  std::uint64_t lastIndex_{0};
  // lane 0 is the queue of the program; the others are the queues of the streams of its kernels
  // (GRAPH_ORIGIN) or helper queues of its own (GRAPH_SPREAD)
  std::vector<Queue*> lanes_;
  // the queue of each lane in submitSpread() (none: the lane runs on the queue of the program)
  std::vector<Queue*> laneQueues_;

  // DEVICE_HSA_QUEUE_SCHEDULE (see submitScheduled()): the levels (packets first..first+count-1),
  // the learned duration of each packet (s, 0: unknown), the signals of a replay that measures them
  struct Level {
    std::size_t first{0};
    std::size_t count{0};
    bool splittable{false};
  };
  std::uint64_t submitScheduled();
  void harvest();
  void ensureLevelSignals(std::size_t level, std::size_t helpers);
  bool schedule_{false};
  std::vector<Level> levels_;
  std::vector<double> estimates_;
  std::vector<std::pair<std::size_t, hsa_signal_t>> measuring_;
  std::uint64_t scheduledReplays_{0};
  std::vector<hsa_signal_t> levelForks_;
  std::vector<std::int64_t> levelForkCounts_;
  std::vector<std::vector<hsa_signal_t>> levelJoins_;
  std::vector<std::vector<std::int64_t>> levelJoinCounts_;

  // DEVICE_HSA_QUEUE_GRAPH_SPREAD=k (see buildSpread()): the work packets each one depends on
  // directly (through nodes without work), and once their durations have been measured, the
  // program spread over the queue and k-1 helper queues of its own
  struct SpreadItem {
    enum class Kind : std::uint8_t { Work, Wait, Fork, Join } kind{Kind::Work};
    // Work: the packet; Wait: the packet of another lane it waits for
    std::size_t index{0};
    unsigned lane{0};
  };
  void buildSpread();
  /// the items of the lanes, the signals between them, for the lane of each packet; the number of
  /// waits on other lanes (durations and predicted ends only for the DAG log)
  std::size_t buildLanes(const std::vector<unsigned>& laneOf,
                         const std::vector<double>& durations,
                         const std::vector<double>& finish);
  std::uint64_t submitSpread();
  void countDownOn(unsigned lane, hsa_signal_t signal);
  unsigned spread_{0};
  // the waits of the lanes (or only those of lane 0 for the others at the end) by kernels that spin
  // on the signal (DEVICE_HSA_QUEUE_GRAPH_SPREAD_WAITS=kernel or join)
  bool spreadKernelWaits_{false};
  bool joinKernelWaits_{false};
  std::unordered_map<std::uint64_t, const std::int64_t*> signalValues_;
  double spreadLatency_{5e-6};
  std::vector<std::vector<std::size_t>> dependencies_;
  std::vector<SpreadItem> spreadItems_;
  std::vector<unsigned> spreadHelpers_;
  // the lane of each packet, its position among the kernels of the lane, the kernels of each lane;
  // the signal of each lane that each of its kernels counts down (see buildLanes())
  std::vector<unsigned> laneOf_;
  std::vector<std::size_t> positions_;
  std::vector<std::size_t> laneKernels_;
  std::vector<hsa_signal_t> laneCounters_;
  hsa_signal_t spreadFork_{};
  std::uint64_t spreadReplays_{0};
};

struct ProgramsImpl {
  std::mutex mutex;
  std::map<Queue*, std::unique_ptr<Program>> byQueue;
  // the HSA queue of the stream whose work each node is (see noteOrigin())
  std::unordered_map<hipGraphNode_t, int> origins;
};

} // namespace device::hsaqueues

#endif // SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_PROGRAM_H_
