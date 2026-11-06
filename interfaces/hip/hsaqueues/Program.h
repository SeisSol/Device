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
  // the lane that runs it: 0 the queue of the program, the others its helper queues
  unsigned lane{0};
  // a fork (the helper lanes of the level start once the lane 0 has reached it) or a join (lane 0
  // goes on once the helper lanes are done with the level), at the given slot of signals
  enum class Kind : std::uint8_t { Work, Fork, Join } kind{Kind::Work};
  std::size_t slot{0};
  std::vector<unsigned> lanes;
  // the level in the graph
  std::size_t level{0};
};

/// the HSA queue number of helper lane `lane` of the programs of queue `queue`
inline int helperQueueNumber(int queue, unsigned lane) {
  return FirstHelperQueue + (queue + 1000) * 16 + static_cast<int>(lane);
}

/**
 * A graph translated for a queue. The kernels of a level of the graph are independent of each
 * other; a queue runs them one after the other though, also without barriers in between (MI250X,
 * ROCm 6.3). With DEVICE_HSA_QUEUE_GRAPH_LANES=k, the levels with at least
 * DEVICE_HSA_QUEUE_GRAPH_LANE_MIN kernels (3 by default) get spread over the queue and k helper
 * queues of its own (lanes), the largest kernels first onto the lane with the least so far; the
 * helpers start the level once the queue has reached it (a fork), and the queue goes on once the
 * helpers are done with it (a join), by signals of the program that each replay counts down. That
 * pays off only with pipes of the command processor left for the helpers (see
 * Runtime::placeQueue()): on the MI250X, with two HSA queues (TPV13 2.16 s instead of 2.74), not
 * with four (2.87 instead of 1.92). With DEVICE_HSA_QUEUE_GRAPH_BORROW=1, the lanes go to other HSA
 * queues instead, those that are idle at each replay (and only while the queue of the program is
 * idle as well, unless DEVICE_HSA_QUEUE_GRAPH_BORROW_IDLE=0); on four queues, that made no
 * difference (1.94 s), and coupled the queues without the condition (2.03 s).
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
      if (packet.kind == ProgramPacket::Kind::Work) {
        text += packet.host ? " h" : " " + std::to_string(packet.level);
      }
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
  // lane 0 is the queue of the program; the others are helper queues of their own, or (borrow_)
  // other HSA queues that are idle when the program gets enqueued, chosen anew each time
  std::vector<Queue*> lanes_;
  bool borrow_{false};
  // the queue of each lane in submit() (none: the lane runs on the queue of the program)
  std::vector<Queue*> laneQueues_;
  std::vector<hsa_signal_t> forks_;
  std::vector<std::vector<hsa_signal_t>> joins_;
  // how often each signal has been counted down so far
  std::vector<std::int64_t> forkCounts_;
  std::vector<std::vector<std::int64_t>> joinCounts_;

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
