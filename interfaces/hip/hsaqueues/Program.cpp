// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Program.h"

#include "Common.h"
#include "Internals.h"
#include "Kernels.h"
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

Programs::Programs() : impl(std::make_unique<ProgramsImpl>()) {}
Programs::~Programs() = default;

namespace {

struct Allocation {
  std::uint64_t begin;
  std::uint64_t end;
  std::uint64_t target;
};

/// moves pointers into the allocations of a graph into the memory that the queue gives them instead
std::uint64_t relocate(const std::vector<Allocation>& allocations, std::uint64_t value) {
  for (const auto& allocation : allocations) {
    if (value >= allocation.begin && value <= allocation.end) {
      return allocation.target + (value - allocation.begin);
    }
  }
  return value;
}

void relocateRange(const std::vector<Allocation>& allocations,
                   std::uint8_t* bytes,
                   std::size_t offset,
                   std::size_t size) {
  if (allocations.empty()) {
    return;
  }
  for (auto position = alignUp(offset, 8); position + 8 <= offset + size; position += 8) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes + position, 8);
    const auto moved = relocate(allocations, value);
    if (moved != value) {
      std::memcpy(bytes + position, &moved, 8);
    }
  }
}

} // namespace

Program::Program(Queue& queue,
                 hipGraph_t graph,
                 const std::unordered_map<hipGraphNode_t, int>& origins)
    : queue_(queue) {
  std::size_t count = 0;
  APIWRAP(hipGraphGetNodes(graph, nullptr, &count));
  std::vector<hipGraphNode_t> nodes(count);
  APIWRAP(hipGraphGetNodes(graph, nodes.data(), &count));
  std::size_t edgeCount = 0;
  APIWRAP(hipGraphGetEdges(graph, nullptr, nullptr, &edgeCount));
  std::vector<hipGraphNode_t> from(edgeCount);
  std::vector<hipGraphNode_t> to(edgeCount);
  APIWRAP(hipGraphGetEdges(graph, from.data(), to.data(), &edgeCount));

  // levels: the longest path from a root; a node starts once the level before has completed
  std::unordered_map<hipGraphNode_t, std::size_t> index;
  for (std::size_t i = 0; i < count; ++i) {
    index[nodes[i]] = i;
  }
  std::vector<std::vector<std::size_t>> successors(count);
  std::vector<std::vector<std::size_t>> predecessors(count);
  std::vector<std::size_t> pending(count, 0);
  for (std::size_t e = 0; e < edgeCount; ++e) {
    successors[index.at(from[e])].push_back(index.at(to[e]));
    predecessors[index.at(to[e])].push_back(index.at(from[e]));
    ++pending[index.at(to[e])];
  }
  std::vector<std::size_t> level(count, 0);
  std::vector<std::size_t> order;
  for (std::size_t i = 0; i < count; ++i) {
    if (pending[i] == 0) {
      order.push_back(i);
    }
  }
  for (std::size_t k = 0; k < order.size(); ++k) {
    for (const auto j : successors[order[k]]) {
      level[j] = std::max(level[j], level[order[k]] + 1);
      if (--pending[j] == 0) {
        order.push_back(j);
      }
    }
  }
  if (order.size() != count) {
    logError() << "HSA queues: a graph has a cycle.";
  }
  // with each node back on the queue of its stream (see below), the lanes run their packets in the
  // order the nodes got added to the graph, as the streams would have (topologically, the earliest
  // added ready node first; HIP lists the nodes in that order): the work of a stream stays
  // together, e.g. that of one cluster, instead of taking turns level by level with the other
  // streams of the queue (TPV5 on one MI300A: 1.78 s by levels)
  const bool byOrigin = setting<bool>("GRAPH_ORIGIN", false) && !origins.empty() &&
                        Runtime::instance().barrierValue && !Runtime::instance().schedule &&
                        setting<unsigned>("GRAPH_LANES", 0U) == 0 &&
                        std::none_of(nodes.begin(), nodes.end(), [](hipGraphNode_t node) {
                          hipGraphNodeType type{};
                          APIWRAP(hipGraphNodeGetType(node, &type));
                          return type == hipGraphNodeTypeHost;
                        });
  if (byOrigin) {
    std::vector<std::size_t> waiting(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
      waiting[i] = predecessors[i].size();
    }
    std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<>> ready;
    for (std::size_t i = 0; i < count; ++i) {
      if (waiting[i] == 0) {
        ready.push(i);
      }
    }
    order.clear();
    while (!ready.empty()) {
      const auto i = ready.top();
      ready.pop();
      order.push_back(i);
      for (const auto j : successors[i]) {
        if (--waiting[j] == 0) {
          ready.push(j);
        }
      }
    }
  } else {
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return level[a] < level[b];
    });
  }

  // the temporary allocations get memory of the queue, one after the other
  std::vector<Allocation> allocations;
  std::uint64_t arenaBytes = 0;
  for (const auto i : order) {
    hipGraphNodeType type{};
    APIWRAP(hipGraphNodeGetType(nodes[i], &type));
    if (type == hipGraphNodeTypeMemAlloc) {
      hipMemAllocNodeParams params{};
      APIWRAP(hipGraphMemAllocNodeGetParams(nodes[i], &params));
      const auto begin = reinterpret_cast<std::uint64_t>(params.dptr);
      allocations.push_back({begin, begin + params.bytesize, arenaBytes});
      arenaBytes = alignUp(arenaBytes + params.bytesize, 256);
    }
  }
  if (!allocations.empty()) {
    if (setting<bool>("VERBOSE", false)) {
      std::set<std::uint64_t> distinct;
      std::uint64_t largest = 0;
      for (const auto& allocation : allocations) {
        distinct.insert(allocation.begin);
        largest = std::max(largest, allocation.end - allocation.begin);
      }
      logInfo() << "HSA queue" << queue.number() << ": a graph allocates" << allocations.size()
                << "times (" << distinct.size() << "distinct addresses, the largest"
                << largest / 1024 << "KiB):" << arenaBytes / (1024 * 1024) << "MiB of the arena";
    }
    std::uint64_t base = 0;
    {
      const std::lock_guard lock(queue.mutex);
      base = reinterpret_cast<std::uint64_t>(queue.arena(arenaBytes));
    }
    for (auto& allocation : allocations) {
      allocation.target += base;
    }
  }

  // the kernels with their arguments (in one block, filled in below), copies, fills, host functions
  struct Pending {
    const KernelInfo* kernel;
    dim3 grid;
    dim3 block;
    std::size_t sharedMemBytes;
    std::vector<std::uint8_t> kernargs;
    std::size_t offset;
    std::function<void()> host;
    std::size_t level;
    // the node of the graph
    std::size_t node;
  };
  std::vector<Pending> work;
  std::size_t kernargBytes = 0;
  std::size_t currentNode = 0;
  const auto addKernel = [&](const KernelInfo& kernel,
                             dim3 grid,
                             dim3 block,
                             std::size_t sharedMemBytes,
                             void** args,
                             std::size_t nodeLevel) {
    Pending item{&kernel, grid, block, sharedMemBytes, {}, 0, {}, nodeLevel, currentNode};
    item.kernargs.resize(std::max<std::size_t>(kernel.kernargSize, 1));
    queue.fillKernargs(kernel, item.kernargs.data(), grid, block, sharedMemBytes, args);
    for (const auto& arg : kernel.args) {
      if (arg.kind == ArgKind::Explicit) {
        relocateRange(allocations, item.kernargs.data(), arg.offset, arg.size);
      }
    }
    item.offset = alignUp(kernargBytes, std::max<std::uint32_t>(kernel.kernargAlign, 16));
    kernargBytes = item.offset + item.kernargs.size();
    work.push_back(std::move(item));
  };

  for (const auto i : order) {
    const auto node = nodes[i];
    currentNode = i;
    hipGraphNodeType type{};
    APIWRAP(hipGraphNodeGetType(node, &type));
    switch (type) {
    case hipGraphNodeTypeKernel: {
      hipKernelNodeParams params{};
      APIWRAP(hipGraphKernelNodeGetParams(node, &params));
      addKernel(Runtime::instance().kernel(params.func),
                params.gridDim,
                params.blockDim,
                params.sharedMemBytes,
                params.kernelParams,
                level[i]);
      break;
    }
    case hipGraphNodeTypeMemcpy: {
      hipMemcpy3DParms params{};
      APIWRAP(hipGraphMemcpyNodeGetParams(node, &params));
      if (params.srcArray != nullptr || params.dstArray != nullptr || params.extent.height > 1 ||
          params.extent.depth > 1 || params.srcPos.x != 0 || params.dstPos.x != 0) {
        logError() << "HSA queues: only plain copies are supported in graphs.";
      }
      auto* dst = reinterpret_cast<void*>(
          relocate(allocations, reinterpret_cast<std::uint64_t>(params.dstPtr.ptr)));
      const auto* src = reinterpret_cast<const void*>(
          relocate(allocations, reinterpret_cast<std::uint64_t>(params.srcPtr.ptr)));
      const auto* dstDevice = deviceAddress(dst);
      const auto* srcDevice = deviceAddress(src);
      if (dstDevice == nullptr || srcDevice == nullptr) {
        logError() << "HSA queues: a graph copies from or to host memory that the device cannot "
                      "access (pageable memory).";
      }
      auto bytes = params.extent.width;
      const auto blocks = static_cast<std::uint32_t>(std::min<std::size_t>(
          MaxCopyBlocks, std::max<std::size_t>(1, bytes / (16 * CopyThreads))));
      void* args[] = {&dstDevice, &srcDevice, &bytes};
      addKernel(Queue::kernelOf(reinterpret_cast<const void*>(hsaCopy)),
                dim3(blocks),
                dim3(CopyThreads),
                0,
                args,
                level[i]);
      break;
    }
    case hipGraphNodeTypeMemset: {
      hipMemsetParams params{};
      APIWRAP(hipGraphMemsetNodeGetParams(node, &params));
      auto* dst = reinterpret_cast<char*>(
          relocate(allocations, reinterpret_cast<std::uint64_t>(params.dst)));
      std::uint32_t value = params.value;
      std::uint32_t elementSize = params.elementSize;
      std::size_t width = params.width;
      std::size_t height = std::max<std::size_t>(params.height, 1);
      std::size_t pitch = params.pitch;
      const auto blocks = static_cast<std::uint32_t>(std::min<std::size_t>(
          MaxCopyBlocks, std::max<std::size_t>(1, width * height / (4 * CopyThreads))));
      void* args[] = {&dst, &value, &elementSize, &width, &height, &pitch};
      addKernel(Queue::kernelOf(reinterpret_cast<const void*>(hsaFill)),
                dim3(blocks),
                dim3(CopyThreads),
                0,
                args,
                level[i]);
      break;
    }
    case hipGraphNodeTypeHost: {
      hipHostNodeParams params{};
      APIWRAP(hipGraphHostNodeGetParams(node, &params));
      Pending item{};
      item.host = [fn = params.fn, data = params.userData]() { fn(data); };
      item.level = level[i];
      item.node = i;
      work.push_back(std::move(item));
      break;
    }
    case hipGraphNodeTypeEmpty:
    case hipGraphNodeTypeMemAlloc:
    case hipGraphNodeTypeMemFree:
      break;
    default:
      logError() << "HSA queues: graph nodes of type" << static_cast<int>(type)
                 << "are not supported.";
    }
  }

  if (kernargBytes > 0) {
    kernargs_ = allocKernargMemory(kernargBytes);
  }
  // the helper lanes, and which level runs on which lanes (see Program)
  schedule_ = Runtime::instance().schedule;
  const auto laneCount =
      Runtime::instance().barrierValue && !schedule_ ? setting<unsigned>("GRAPH_LANES", 0U) : 0U;
  const auto laneMinimum = setting<std::size_t>("GRAPH_LANE_MIN", std::size_t{3});
  borrow_ = laneCount > 0 && setting<bool>("GRAPH_BORROW", false);
  lanes_.push_back(&queue);
  for (unsigned lane = 1; lane <= laneCount; ++lane) {
    lanes_.push_back(borrow_ ? nullptr
                             : &Runtime::instance().queue(helperQueueNumber(queue.number(), lane)));
  }
  std::vector<unsigned> laneOf(work.size(), 0);
  std::vector<bool> splitLevel(work.size(), false);
  std::size_t splitLevels = 0;
  for (std::size_t first = 0; first < work.size();) {
    auto last = first;
    bool host = false;
    while (last < work.size() && work[last].level == work[first].level) {
      host = host || static_cast<bool>(work[last].host);
      ++last;
    }
    if (laneCount > 0 && !host && last - first >= laneMinimum) {
      // the largest first onto the lane with the least so far (the queue first): the kernels are
      // small, so each one costs about the same, and more so with many blocks
      std::vector<std::size_t> items(last - first);
      for (std::size_t k = 0; k < items.size(); ++k) {
        items[k] = first + k;
      }
      const auto weight = [&](std::size_t k) {
        return 128 + static_cast<std::uint64_t>(work[k].grid.x) * work[k].grid.y * work[k].grid.z;
      };
      std::stable_sort(items.begin(), items.end(), [&](std::size_t a, std::size_t b) {
        return weight(a) > weight(b);
      });
      std::vector<std::uint64_t> load(laneCount + 1, 0);
      for (const auto k : items) {
        unsigned best = 0;
        for (unsigned lane = 1; lane <= laneCount; ++lane) {
          if (load[lane] < load[best]) {
            best = lane;
          }
        }
        laneOf[k] = best;
        load[best] += weight(k);
      }
      for (auto k = first; k < last; ++k) {
        splitLevel[k] = true;
      }
      ++splitLevels;
    }
    first = last;
  }
  std::size_t previousLevel = SIZE_MAX;
  for (std::size_t index = 0; index < work.size(); ++index) {
    auto& item = work[index];
    const bool levelStart = item.level != previousLevel;
    if (levelStart && splitLevel[index]) {
      // the lanes of this level
      std::vector<unsigned> used;
      for (auto k = index; k < work.size() && work[k].level == item.level; ++k) {
        if (laneOf[k] != 0 && std::find(used.begin(), used.end(), laneOf[k]) == used.end()) {
          used.push_back(laneOf[k]);
        }
      }
      std::sort(used.begin(), used.end());
      ProgramPacket fork;
      fork.kind = ProgramPacket::Kind::Fork;
      fork.slot = forks_.size();
      fork.lanes = used;
      hsa_signal_t signal{};
      check(hsa_amd_signal_create(CounterStart, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal),
            "hsa_amd_signal_create");
      forks_.push_back(signal);
      forkCounts_.push_back(0);
      std::vector<hsa_signal_t> joins;
      for (std::size_t k = 0; k < used.size(); ++k) {
        check(hsa_amd_signal_create(CounterStart, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal),
              "hsa_amd_signal_create");
        joins.push_back(signal);
      }
      joins_.push_back(joins);
      joinCounts_.emplace_back(used.size(), 0);
      packets_.push_back(std::move(fork));
    }
    ProgramPacket packet;
    packet.lane = laneOf[index];
    packet.level = item.level;
    // the first one of each level waits for all before it (and so for the work before the graph);
    // on a helper lane, the fork has waited
    const bool barrier = levelStart || packet.lane != 0 || splitLevel[index];
    previousLevel = item.level;
    if (item.host) {
      packet.host = std::move(item.host);
      // a host function waits for all before it, and all after it wait for it
      previousLevel = SIZE_MAX;
    } else {
      auto* kernargs = static_cast<std::uint8_t*>(kernargs_) + item.offset;
      std::memcpy(kernargs, item.kernargs.data(), item.kernargs.size());
      auto& dispatch = packet.packet;
      const auto& kernel = *item.kernel;
      dispatch.workgroup_size_x = static_cast<std::uint16_t>(item.block.x);
      dispatch.workgroup_size_y = static_cast<std::uint16_t>(item.block.y);
      dispatch.workgroup_size_z = static_cast<std::uint16_t>(item.block.z);
      dispatch.grid_size_x = item.grid.x * item.block.x;
      dispatch.grid_size_y = item.grid.y * item.block.y;
      dispatch.grid_size_z = item.grid.z * item.block.z;
      dispatch.private_segment_size =
          kernel.dynamicStack ? std::max(kernel.privateSize, Runtime::instance().stackSize)
                              : kernel.privateSize;
      dispatch.group_segment_size =
          kernel.groupSize + static_cast<std::uint32_t>(item.sharedMemBytes);
      dispatch.kernel_object = kernel.object;
      dispatch.kernarg_address = kernargs;
      packet.header =
          static_cast<std::uint16_t>((HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                                     (barrier ? BarrierBit : 0) | KernelFences);
      packet.rest = static_cast<std::uint16_t>(3U << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
      if (kernel.unsupported) {
        warnOnce("the kernel " + kernel.name +
                 " may use printf, hostcalls, the device heap or device enqueue, which HSA "
                 "queues do not provide");
      }
    }
    packets_.push_back(std::move(packet));
    const bool levelEnd = index + 1 == work.size() || work[index + 1].level != item.level;
    if (levelEnd && splitLevel[index]) {
      ProgramPacket join;
      join.kind = ProgramPacket::Kind::Join;
      join.slot = forks_.size() - 1;
      for (const auto& p : packets_) {
        if (p.kind == ProgramPacket::Kind::Fork && p.slot == join.slot) {
          join.lanes = p.lanes;
        }
      }
      packets_.push_back(std::move(join));
    }
  }
  if (kernargBytes > 0) {
    flushKernargs(static_cast<const char*>(kernargs_) + kernargBytes - 1);
  }
  if (schedule_) {
    // (no fixed lanes: only work packets)
    for (std::size_t first = 0; first < packets_.size();) {
      auto last = first;
      bool host = false;
      while (last < packets_.size() && packets_[last].level == packets_[first].level) {
        host = host || static_cast<bool>(packets_[last].host);
        ++last;
      }
      levels_.push_back({first, last - first, !host && last - first >= 2});
      first = last;
    }
    estimates_.assign(packets_.size(), 0.0);
    levelForks_.resize(levels_.size());
    levelForkCounts_.assign(levels_.size(), 0);
    levelJoins_.resize(levels_.size());
    levelJoinCounts_.resize(levels_.size());
  }
  // the work packets each one waits for directly, through the nodes without work
  const auto findDependencies = [&]() {
    std::vector<std::ptrdiff_t> workOfNode(count, -1);
    for (std::size_t k = 0; k < work.size(); ++k) {
      workOfNode[work[k].node] = static_cast<std::ptrdiff_t>(k);
    }
    std::vector<std::vector<std::size_t>> before(count);
    for (const auto i : order) {
      std::vector<std::size_t> found;
      for (const auto p : predecessors[i]) {
        if (workOfNode[p] >= 0) {
          found.push_back(static_cast<std::size_t>(workOfNode[p]));
        } else {
          found.insert(found.end(), before[p].begin(), before[p].end());
        }
      }
      std::sort(found.begin(), found.end());
      found.erase(std::unique(found.begin(), found.end()), found.end());
      before[i] = std::move(found);
    }
    dependencies_.resize(work.size());
    for (std::size_t k = 0; k < work.size(); ++k) {
      dependencies_[k] = before[work[k].node];
    }
  };
  // lanes of their own only for programs of kernels (a host function would wait for all lanes), not
  // with the lanes of levels or the schedule
  const bool anyHost = std::any_of(
      work.begin(), work.end(), [](const Pending& item) { return static_cast<bool>(item.host); });
  const bool ownLanes =
      Runtime::instance().barrierValue && laneCount == 0 && !schedule_ && !anyHost;
  const auto waits = setting<std::string>("GRAPH_SPREAD_WAITS", "packet");
  spreadKernelWaits_ = waits == "kernel";
  joinKernelWaits_ = waits == "kernel" || waits == "join";
  // each packet back on the queue of the stream whose work it is (see noteOrigin())
  if (byOrigin && ownLanes) {
    findDependencies();
    // the queues with the most work get lanes of their own, up to
    // DEVICE_HSA_QUEUE_GRAPH_ORIGIN_LANES with that of the program; the others, and the queue of
    // the program, run on lane 0
    const auto maxLanes = setting<std::size_t>("GRAPH_ORIGIN_LANES", std::size_t{0});
    std::vector<int> originOf(work.size(), std::numeric_limits<int>::min());
    std::map<int, std::size_t> workOf;
    for (std::size_t k = 0; k < work.size(); ++k) {
      if (const auto found = origins.find(nodes[work[k].node]); found != origins.end()) {
        originOf[k] = found->second;
        ++workOf[found->second];
      }
    }
    std::vector<std::pair<std::size_t, int>> byWork;
    for (const auto& [number, packets] : workOf) {
      if (number != queue.number()) {
        byWork.emplace_back(packets, number);
      }
    }
    std::stable_sort(byWork.begin(), byWork.end(), [](const auto& a, const auto& b) {
      return a.first > b.first;
    });
    std::map<int, unsigned> laneOfQueue{{queue.number(), 0U}};
    for (const auto& [packets, number] : byWork) {
      if (maxLanes == 0 || lanes_.size() < maxLanes) {
        laneOfQueue[number] = static_cast<unsigned>(lanes_.size());
        lanes_.push_back(&Runtime::instance().queue(number));
      } else {
        laneOfQueue[number] = 0;
      }
    }
    // (work of streams that are not HSA streams goes with the first packet it depends on)
    std::vector<unsigned> laneOf(work.size(), 0);
    for (std::size_t k = 0; k < work.size(); ++k) {
      if (originOf[k] != std::numeric_limits<int>::min()) {
        laneOf[k] = laneOfQueue.at(originOf[k]);
      } else if (!dependencies_[k].empty()) {
        laneOf[k] = laneOf[dependencies_[k].front()];
      }
    }
    if (lanes_.size() >= 2) {
      spread_ = static_cast<unsigned>(lanes_.size());
      const auto waitCount = buildLanes(
          laneOf, std::vector<double>(work.size(), 0.0), std::vector<double>(work.size(), 0.0));
      if (setting<bool>("VERBOSE", false)) {
        std::string text;
        for (const auto& [number, lane] : laneOfQueue) {
          text += " " + std::to_string(number) + "->" + std::to_string(lane);
        }
        logInfo() << "HSA queue" << queue.number() << ": a graph of" << work.size()
                  << "kernels back on the queues of their streams (queue->lane:" << text.c_str()
                  << "), with" << waitCount << "waits on other lanes";
      }
    } else {
      lanes_.resize(1);
    }
  }
  // spread over helper queues once the durations are known (see buildSpread())
  const auto spread = setting<unsigned>("GRAPH_SPREAD", 0U);
  if (spread_ == 0 && spread >= 2 && ownLanes &&
      work.size() >= setting<std::size_t>("GRAPH_SPREAD_MIN", std::size_t{32})) {
    spread_ = spread;
    spreadLatency_ = setting<double>("GRAPH_SPREAD_LATENCY_US", 5.0) * 1e-6;
    findDependencies();
    estimates_.assign(packets_.size(), 0.0);
    for (unsigned lane = 1; lane < spread_; ++lane) {
      lanes_.push_back(&Runtime::instance().queue(helperQueueNumber(queue.number(), lane)));
    }
    const std::lock_guard lock(queue.mutex);
    queue.enableProfiling();
  }
  if (setting<bool>("VERBOSE", false)) {
    std::map<std::size_t, std::size_t> kernelsPerLevel;
    for (const auto& item : work) {
      if (!item.host) {
        ++kernelsPerLevel[item.level];
      }
    }
    std::size_t widest = 0;
    for (const auto& [lvl, kernels] : kernelsPerLevel) {
      widest = std::max(widest, kernels);
    }
    logInfo() << "HSA queue" << queue.number() << ": a graph of" << count << "nodes and"
              << edgeCount << "edges became" << work.size() << "packets in"
              << kernelsPerLevel.size() << "levels of up to" << widest << "kernels;" << splitLevels
              << "of them spread over" << laneCount
              << (borrow_ ? "other HSA queues while idle" : "helper queues");
  }
}

Program::~Program() {
  if (!enabled()) {
    // after finalize(), the queue is gone (graphs may outlive the device)
    return;
  }
  std::uint64_t mark = 0;
  {
    const std::lock_guard lock(queue_.mutex);
    mark = queue_.markCoveringLocked(lastIndex_);
  }
  queue_.wait(mark);
  if (kernargs_ != nullptr) {
    hsa_amd_memory_pool_free(kernargs_);
  }
  // (the queue has waited for the helpers at each join)
  for (auto& [index, signal] : measuring_) {
    Runtime::instance().freeTraceSignal(signal);
  }
  for (const auto signal : levelForks_) {
    if (signal.handle != 0) {
      hsa_signal_destroy(signal);
    }
  }
  for (const auto& joins : levelJoins_) {
    for (const auto signal : joins) {
      hsa_signal_destroy(signal);
    }
  }
  for (const auto signal : forks_) {
    hsa_signal_destroy(signal);
  }
  for (const auto& joins : joins_) {
    for (const auto signal : joins) {
      hsa_signal_destroy(signal);
    }
  }
  // (the queue has waited for the lanes at the join of the last replay)
  for (const auto signal : laneCounters_) {
    if (signal.handle != 0) {
      hsa_signal_destroy(signal);
    }
  }
  if (spreadFork_.handle != 0) {
    hsa_signal_destroy(spreadFork_);
  }
}

void Program::push(unsigned lane, const PacketImage& image) {
  auto* queue = laneQueues_[lane];
  if (queue == nullptr || queue == &queue_) {
    queue_.push(image);
    return;
  }
  if (borrow_) {
    // (locked in submit())
    queue->push(image);
    return;
  }
  // (the queue's mutex is held; the helpers of a queue are only ever locked after it)
  const std::lock_guard lock(queue->mutex);
  queue->push(image);
}

void Program::waitOnLanes(hsa_signal_t signal,
                          std::int64_t count,
                          const std::vector<unsigned>& lanes,
                          bool byKernel) {
  // it has counted down `count` times once it is below the start minus the times before
  const auto below = CounterStart - count + 1;
  for (const auto lane : lanes) {
    if (byKernel) {
      // a wave that waits instead of the command processor (see
      // DEVICE_HSA_QUEUE_GRAPH_SPREAD_WAITS)
      auto& value = signalValues_[signal.handle];
      if (value == nullptr) {
        volatile hsa_signal_value_t* pointer = nullptr;
        check(hsa_amd_signal_value_pointer(signal, &pointer), "hsa_amd_signal_value_pointer");
        value = const_cast<const std::int64_t*>(pointer);
      }
      auto* queue = laneQueues_[lane] == nullptr ? &queue_ : laneQueues_[lane];
      auto argument = value;
      auto bound = below;
      void* args[] = {&argument, &bound};
      const auto& kernel =
          Runtime::instance().kernel(reinterpret_cast<const void*>(hsaHipWaitBelow));
      if (queue == &queue_) {
        queue->dispatch(kernel, dim3(1), dim3(1), 0, args, QueueWaitFences, hsa_signal_t{0});
      } else {
        const std::lock_guard lock(queue->mutex);
        queue->dispatch(kernel, dim3(1), dim3(1), 0, args, QueueWaitFences, hsa_signal_t{0});
      }
      continue;
    }
    PacketImage packet;
    auto& barrier = packet.as<hsa_amd_barrier_value_packet_t>();
    barrier.signal = signal;
    barrier.value = below;
    barrier.mask = -1;
    barrier.cond = HSA_SIGNAL_CONDITION_LT;
    packet.setHeader(
        static_cast<std::uint16_t>((HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) |
                                   BarrierBit | QueueWaitFences),
        HSA_AMD_PACKET_TYPE_BARRIER_VALUE);
    push(lane, packet);
  }
}

std::uint64_t Program::submit() {
  if (schedule_) {
    return submitScheduled();
  }
  // spread: the first replays run on the queue, the first one of them with the times of its
  // kernels taken; once they are there, the program gets spread
  bool measure = false;
  if (spread_ >= 2) {
    harvest();
    if (spreadItems_.empty() && measuring_.empty() && spreadReplays_ > 0) {
      buildSpread();
    }
    if (!spreadItems_.empty()) {
      return submitSpread();
    }
    measure = measuring_.empty() && spreadReplays_ == 0;
    if (measure) {
      ++spreadReplays_;
    }
  }
  laneQueues_.assign(lanes_.begin(), lanes_.end());
  std::vector<std::unique_lock<std::mutex>> borrowed;
  static std::atomic<std::uint64_t> submits{0};
  static std::atomic<std::uint64_t> lent{0};
  static const bool borrowWhenIdle = setting<bool>("GRAPH_BORROW_IDLE", true);
  static const bool verbose = setting<bool>("VERBOSE", false);
  const bool borrowNow = borrow_ && (!borrowWhenIdle || queue_.idleLocked());
  if (borrowNow) {
    // other HSA queues on other pipes that have nothing to do (with the queue of the program idle
    // as well: the level starts right away, and the helpers hold up the work of their own queue as
    // little as they can); a lane without one runs on the queue of the program (its mutex is held:
    // the others only get tried, never waited for)
    const auto candidates = Runtime::instance().borrowable(queue_);
    std::size_t next = 0;
    for (std::size_t lane = 1; lane < laneQueues_.size(); ++lane) {
      while (next < candidates.size()) {
        auto* candidate = candidates[next++];
        std::unique_lock lock(candidate->mutex, std::try_to_lock);
        if (lock.owns_lock() && candidate->idleLocked()) {
          laneQueues_[lane] = candidate;
          borrowed.push_back(std::move(lock));
          break;
        }
      }
    }
  }
  if (borrow_ && verbose) {
    const auto count = ++submits;
    lent += borrowed.size();
    if (count % 20000 == 0) {
      logInfo() << "HSA queues: graphs borrowed" << lent.load() << "lanes in" << count << "replays";
    }
  }
  constexpr std::uint16_t LaneFences = fenceBits(HSA_FENCE_SCOPE_NONE, HSA_FENCE_SCOPE_AGENT);
  const auto countDown = [&](unsigned lane, hsa_signal_t signal) {
    PacketImage packet;
    packet.as<hsa_barrier_and_packet_t>().completion_signal = signal;
    packet.setHeader(
        static_cast<std::uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                   BarrierBit | LaneFences),
        0);
    push(lane, packet);
  };
  for (const auto& packet : packets_) {
    if (packet.kind == ProgramPacket::Kind::Fork) {
      // the queue has done all before the level; then the helpers start it (the lanes on the
      // queue of the program run in its order)
      std::vector<unsigned> lanes;
      for (const auto lane : packet.lanes) {
        if (laneQueues_[lane] != nullptr) {
          lanes.push_back(lane);
        }
      }
      if (!lanes.empty()) {
        countDown(0, forks_[packet.slot]);
        waitOnLanes(forks_[packet.slot], ++forkCounts_[packet.slot], lanes);
      }
    } else if (packet.kind == ProgramPacket::Kind::Join) {
      // the helpers are done with the level; the queue waits for them
      for (std::size_t k = 0; k < packet.lanes.size(); ++k) {
        if (laneQueues_[packet.lanes[k]] != nullptr) {
          countDown(packet.lanes[k], joins_[packet.slot][k]);
          waitOnLanes(joins_[packet.slot][k], ++joinCounts_[packet.slot][k], {0});
        }
      }
    } else if (packet.host) {
      queue_.hostFunction(packet.host);
    } else {
      PacketImage image;
      image.as<hsa_kernel_dispatch_packet_t>() = packet.packet;
      if (measure) {
        const auto signal = Runtime::instance().traceSignal();
        image.as<hsa_kernel_dispatch_packet_t>().completion_signal = signal;
        measuring_.emplace_back(static_cast<std::size_t>(&packet - packets_.data()), signal);
      }
      image.setHeader(packet.header, packet.rest);
      push(packet.lane, image);
    }
  }
  lastIndex_ = queue_.created();
  return lastIndex_;
}

void Program::countDownOn(unsigned lane, hsa_signal_t signal) {
  constexpr std::uint16_t LaneFences = fenceBits(HSA_FENCE_SCOPE_NONE, HSA_FENCE_SCOPE_AGENT);
  PacketImage packet;
  packet.as<hsa_barrier_and_packet_t>().completion_signal = signal;
  packet.setHeader(
      static_cast<std::uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                 BarrierBit | LaneFences),
      0);
  push(lane, packet);
}

/**
 * Spreads the program over its queue and spread_ - 1 helper queues of its own, once the durations
 * of its packets have been measured: each packet, in the order of the program, goes to the lane
 * where it would end first (EFT), with spreadLatency_ for each packet it depends on on another
 * lane; so a packet stays on the lane of what it waits for, unless another lane is free earlier.
 * Each lane runs its packets one after the other (barrier bits). Before a packet, its lane waits
 * for the latest packet it depends on of each other lane (a barrier-value packet on a signal that
 * the packet counts down, unless the lane has waited for that one or a later one of it before). The
 * helper lanes start once the queue has done the work before the program (a fork), and the queue
 * goes on once they are done (a join). The graph of a super-timestep of SeisSol on one MI300A
 * (DAG logs of TPV5 and TPV13, simulated): one queue takes the work of all of it (2.4 and 3.9 s),
 * the queues it came from 1.24 and 2.13 s, four lanes this way 1.05 and 1.54 s, its longest path.
 */
void Program::buildSpread() {
  auto& runtime = Runtime::instance();
  const auto n = packets_.size();
  std::vector<double> finish(n, 0.0);
  std::vector<unsigned> laneOf(n, 0);
  std::vector<double> laneFree(spread_, 0.0);
  std::vector<double> durations(n, 0.0);
  double work = 0;
  for (std::size_t k = 0; k < n; ++k) {
    auto duration = estimates_[k];
    if (duration <= 0) {
      duration = runtime.kernelEstimate(packets_[k].packet.kernel_object);
    }
    if (duration <= 0) {
      duration = 10e-6;
    }
    durations[k] = duration;
    work += duration;
    unsigned best = 0;
    double bestEnd = std::numeric_limits<double>::infinity();
    for (unsigned lane = 0; lane < spread_; ++lane) {
      double ready = laneFree[lane];
      for (const auto p : dependencies_[k]) {
        ready = std::max(ready, finish[p] + (laneOf[p] != lane ? spreadLatency_ : 0.0));
      }
      if (ready + duration < bestEnd) {
        bestEnd = ready + duration;
        best = lane;
      }
    }
    laneOf[k] = best;
    finish[k] = bestEnd;
    laneFree[best] = bestEnd;
  }
  const auto waits = buildLanes(laneOf, durations, finish);
  spreadReplays_ = 0;
  if (setting<bool>("VERBOSE", false)) {
    std::vector<std::size_t> sizes(spread_, 0);
    for (std::size_t k = 0; k < n; ++k) {
      ++sizes[laneOf[k]];
    }
    std::string lanes;
    for (unsigned lane = 0; lane < spread_; ++lane) {
      lanes += " " + std::to_string(sizes[lane]);
    }
    logInfo() << "HSA queue" << queue_.number() << ": a graph of" << n << "kernels spread over"
              << spread_ << "lanes (kernels:" << lanes.c_str() << "), with" << waits
              << "waits on other lanes; predicted"
              << *std::max_element(finish.begin(), finish.end()) * 1e3 << "ms for" << work * 1e3
              << "ms of work";
  }
}

/**
 * Each lane runs its packets one after the other (barrier bits), and each of its kernels counts
 * down a signal of the lane. Before a packet, its lane waits for the latest packet it depends on of
 * each other lane (a barrier-value packet on the signal of that lane, unless the lane has waited
 * for that one or a later one of it before). The other lanes start once the queue has done the work
 * before the program (a fork), and the queue goes on once they are done (a join).
 *
 * The waits are on signals that keep changing while the lanes work: on the MI300A, a queue that
 * waits in the command processor for a signal that does not change slows down the kernels of the
 * other queues (one such queue next to SeisSol: 43-56 % longer), one that waits for a signal that
 * changes with each kernel of another queue hardly does. So before the fork, a lane waits for the
 * other lanes to be done with the replay before (as the fork would), and the queue waits for the
 * lanes by their signals as well.
 */
std::size_t Program::buildLanes(const std::vector<unsigned>& laneOf,
                                const std::vector<double>& durations,
                                const std::vector<double>& finish) {
  auto& runtime = Runtime::instance();
  const auto n = packets_.size();
  laneOf_ = laneOf;
  positions_.assign(n, 0);
  laneKernels_.assign(spread_, 0);
  for (std::size_t k = 0; k < n; ++k) {
    positions_[k] = laneKernels_[laneOf[k]]++;
  }
  for (unsigned lane = 1; lane < spread_; ++lane) {
    if (laneKernels_[lane] > 0) {
      spreadHelpers_.push_back(lane);
    }
  }
  // the latest position of each other lane that a lane has waited for
  std::vector<std::vector<std::ptrdiff_t>> waited(spread_,
                                                  std::vector<std::ptrdiff_t>(spread_, -1));
  std::size_t waits = 0;
  spreadItems_.push_back({SpreadItem::Kind::Fork});
  for (std::size_t k = 0; k < n; ++k) {
    const auto lane = laneOf[k];
    std::vector<std::ptrdiff_t> latest(spread_, -1);
    for (const auto p : dependencies_[k]) {
      if (laneOf[p] != lane) {
        latest[laneOf[p]] = std::max(latest[laneOf[p]], static_cast<std::ptrdiff_t>(p));
      }
    }
    for (unsigned other = 0; other < spread_; ++other) {
      const auto p = latest[other];
      if (p < 0 || static_cast<std::ptrdiff_t>(positions_[p]) <= waited[lane][other]) {
        continue;
      }
      SpreadItem wait;
      wait.kind = SpreadItem::Kind::Wait;
      wait.index = static_cast<std::size_t>(p);
      wait.lane = lane;
      spreadItems_.push_back(wait);
      waited[lane][other] = static_cast<std::ptrdiff_t>(positions_[p]);
      ++waits;
    }
    SpreadItem item;
    item.index = k;
    item.lane = lane;
    spreadItems_.push_back(item);
  }
  spreadItems_.push_back({SpreadItem::Kind::Join});
  // (each signal gets counted down by one queue only)
  const auto newSignal = []() {
    hsa_signal_t signal{};
    check(hsa_amd_signal_create(CounterStart, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal),
          "hsa_amd_signal_create");
    return signal;
  };
  for (unsigned lane = 0; lane < spread_; ++lane) {
    laneCounters_.push_back(laneKernels_[lane] > 0 ? newSignal() : hsa_signal_t{});
  }
  spreadFork_ = newSignal();
  if (dagLog().on()) {
    // Y program packet lane duration finish name dependencies...; X program kind lane index
    for (std::size_t k = 0; k < n; ++k) {
      std::string dependencies;
      for (const auto p : dependencies_[k]) {
        dependencies += " " + std::to_string(p);
      }
      dagLog().line("Y %p %zu %u %lld %lld %s%s\n",
                    static_cast<void*>(this),
                    k,
                    laneOf[k],
                    static_cast<long long>(durations[k] * 1e9),
                    static_cast<long long>(finish[k] * 1e9),
                    runtime.kernelName(packets_[k].packet.kernel_object).c_str(),
                    dependencies.c_str());
    }
    for (const auto& item : spreadItems_) {
      dagLog().line("X %p %d %u %zu\n",
                    static_cast<void*>(this),
                    static_cast<int>(item.kind),
                    item.lane,
                    item.index);
    }
  }
  return waits;
}

std::uint64_t Program::submitSpread() {
  const auto count = static_cast<std::int64_t>(++spreadReplays_);
  laneQueues_.assign(lanes_.begin(), lanes_.end());
  // (for the DAG log: where the packets of each lane begin on its queue)
  std::vector<std::uint64_t> firsts;
  const bool logged = dagLog().on();
  if (logged) {
    for (auto* queue : laneQueues_) {
      if (queue == nullptr || queue == &queue_) {
        firsts.push_back(queue_.created());
      } else {
        const std::lock_guard lock(queue->mutex);
        firsts.push_back(queue->created());
      }
    }
  }
  const bool tracing = Runtime::instance().tracing.load(std::memory_order_relaxed);
  const auto replay = static_cast<std::size_t>(count);
  for (const auto& item : spreadItems_) {
    switch (item.kind) {
    case SpreadItem::Kind::Fork:
      if (!spreadHelpers_.empty()) {
        countDownOn(0, spreadFork_);
        for (const auto helper : spreadHelpers_) {
          // (while the others are still busy with the replay before; see buildLanes())
          for (unsigned lane = 0; replay > 1 && lane < spread_; ++lane) {
            if (lane != helper && laneKernels_[lane] > 0) {
              waitOnLanes(laneCounters_[lane],
                          static_cast<std::int64_t>((replay - 1) * laneKernels_[lane]),
                          {helper},
                          spreadKernelWaits_);
            }
          }
          waitOnLanes(spreadFork_, count, {helper}, spreadKernelWaits_);
        }
      }
      break;
    case SpreadItem::Kind::Wait: {
      const auto other = laneOf_[item.index];
      waitOnLanes(laneCounters_[other],
                  static_cast<std::int64_t>((replay - 1) * laneKernels_[other] +
                                            positions_[item.index] + 1),
                  {item.lane},
                  spreadKernelWaits_);
      break;
    }
    case SpreadItem::Kind::Work: {
      const auto& packet = packets_[item.index];
      PacketImage image;
      image.as<hsa_kernel_dispatch_packet_t>() = packet.packet;
      if (!tracing) {
        image.as<hsa_kernel_dispatch_packet_t>().completion_signal = laneCounters_[item.lane];
      }
      image.setHeader(static_cast<std::uint16_t>(packet.header | BarrierBit), packet.rest);
      push(item.lane, image);
      if (tracing) {
        // (the kernel gets a signal of the trace instead)
        countDownOn(item.lane, laneCounters_[item.lane]);
      }
      break;
    }
    case SpreadItem::Kind::Join:
      for (const auto helper : spreadHelpers_) {
        waitOnLanes(laneCounters_[helper],
                    static_cast<std::int64_t>(replay * laneKernels_[helper]),
                    {0},
                    joinKernelWaits_);
      }
      break;
    }
  }
  if (logged) {
    // Z program lane queue first last: the packets first+1..last of a lane in this replay
    for (std::size_t lane = 0; lane < laneQueues_.size(); ++lane) {
      auto* queue = laneQueues_[lane] == nullptr ? &queue_ : laneQueues_[lane];
      std::uint64_t last = 0;
      if (queue == &queue_) {
        last = queue_.created();
      } else {
        const std::lock_guard lock(queue->mutex);
        last = queue->created();
      }
      dagLog().line("Z %p %zu %d %llu %llu\n",
                    static_cast<void*>(this),
                    lane,
                    queue->number(),
                    static_cast<unsigned long long>(firsts[lane]),
                    static_cast<unsigned long long>(last));
    }
  }
  lastIndex_ = queue_.created();
  return lastIndex_;
}

void Program::harvest() {
  if (measuring_.empty()) {
    return;
  }
  for (const auto& [index, signal] : measuring_) {
    if (hsa_signal_load_relaxed(signal) > 0) {
      // (not done yet)
      return;
    }
  }
  auto& runtime = Runtime::instance();
  for (const auto& [index, signal] : measuring_) {
    hsa_amd_profiling_dispatch_time_t time{};
    if (hsa_amd_profiling_get_dispatch_time(runtime.gpu, signal, &time) == HSA_STATUS_SUCCESS &&
        time.end > time.start) {
      const double seconds = static_cast<double>(time.end - time.start) /
                             static_cast<double>(runtime.timestampFrequency);
      auto& estimate = estimates_[index];
      estimate = estimate <= 0 ? seconds : 0.75 * estimate + 0.25 * seconds;
      runtime.learnKernel(packets_[index].packet.kernel_object, seconds);
    }
    runtime.freeTraceSignal(signal);
  }
  measuring_.clear();
}

// (a join signal for each helper: one that several queues count down loses decrements on gfx90a,
// ROCm 6.3, by barrier packets and kernels alike: 4 queues of 2000 packets each lost 900 to 3600)
void Program::ensureLevelSignals(std::size_t level, std::size_t helpers) {
  hsa_signal_t signal{};
  if (levelForks_[level].handle == 0) {
    check(hsa_amd_signal_create(CounterStart, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal),
          "hsa_amd_signal_create");
    levelForks_[level] = signal;
  }
  while (levelJoins_[level].size() < helpers) {
    check(hsa_amd_signal_create(CounterStart, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal),
          "hsa_amd_signal_create");
    levelJoins_[level].push_back(signal);
    levelJoinCounts_[level].push_back(0);
  }
}

/**
 * Enqueues the program level by level (DEVICE_HSA_QUEUE_SCHEDULE): a level of independent kernels
 * goes onto the queue of the program, or, where that ends it earlier, partly onto HSA queues on
 * other pipes (the queues of a pipe run one kernel after the other): the longest kernel first onto
 * the queue that ends it first, by the durations the program has measured and the predicted work of
 * the queues, with a fork (the others start the level once the queue of the program has reached it)
 * and a join (the queue of the program goes on once they are done). The other queues only get
 * tried, never waited for (the mutex of the queue of the program is held); their work after this
 * waits for it.
 */
std::uint64_t Program::submitScheduled() {
  auto& runtime = Runtime::instance();
  harvest();
  ++scheduledReplays_;
  // a few replays early on, then every 128th, take the times of all kernels
  const bool measure =
      measuring_.empty() &&
      (scheduledReplays_ <= 4 ? scheduledReplays_ >= 2 : scheduledReplays_ % 128 == 0);
  std::vector<Queue*> helpers;
  std::vector<std::unique_lock<std::mutex>> locks;
  for (auto* candidate : runtime.borrowable(queue_)) {
    std::unique_lock lock(candidate->mutex, std::try_to_lock);
    if (lock.owns_lock()) {
      helpers.push_back(candidate);
      locks.push_back(std::move(lock));
    }
  }
  const double now = steadySeconds();
  double time = queue_.predictedFreeLocked(now);
  std::vector<double> helperFree(helpers.size());
  for (std::size_t k = 0; k < helpers.size(); ++k) {
    helperFree[k] = helpers[k]->predictedFreeLocked(now);
  }
  std::vector<bool> helperUsed(helpers.size(), false);

  constexpr std::uint16_t LaneFences = fenceBits(HSA_FENCE_SCOPE_NONE, HSA_FENCE_SCOPE_AGENT);
  const auto countDown = [&](Queue& target, hsa_signal_t signal) {
    PacketImage packet;
    packet.as<hsa_barrier_and_packet_t>().completion_signal = signal;
    packet.setHeader(
        static_cast<std::uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                   BarrierBit | LaneFences),
        0);
    target.push(packet);
  };
  const auto waitCount = [&](Queue& target, hsa_signal_t signal, std::int64_t count) {
    PacketImage packet;
    auto& barrier = packet.as<hsa_amd_barrier_value_packet_t>();
    barrier.signal = signal;
    barrier.value = CounterStart - count + 1;
    barrier.mask = -1;
    barrier.cond = HSA_SIGNAL_CONDITION_LT;
    packet.setHeader(
        static_cast<std::uint16_t>((HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) |
                                   BarrierBit | QueueWaitFences),
        HSA_AMD_PACKET_TYPE_BARRIER_VALUE);
    target.push(packet);
  };
  const auto emit = [&](Queue& target, std::size_t index, bool barrier) {
    const auto& packet = packets_[index];
    if (packet.host) {
      queue_.hostFunction(packet.host);
      return;
    }
    PacketImage image;
    image.as<hsa_kernel_dispatch_packet_t>() = packet.packet;
    if (measure) {
      const auto signal = runtime.traceSignal();
      image.as<hsa_kernel_dispatch_packet_t>().completion_signal = signal;
      measuring_.emplace_back(index, signal);
    }
    image.setHeader(barrier ? static_cast<std::uint16_t>(packet.header | BarrierBit)
                            : packet.header,
                    packet.rest);
    target.push(image);
  };

  std::size_t splitLevels = 0;
  std::size_t movedKernels = 0;
  for (std::size_t l = 0; l < levels_.size(); ++l) {
    const auto& level = levels_[l];
    double work = 0;
    bool known = true;
    bool small = true;
    for (std::size_t i = 0; i < level.count; ++i) {
      const auto estimate = estimates_[level.first + i];
      known = known && estimate > 0;
      work += estimate > 0 ? estimate : DefaultKernelSeconds;
      const auto& dispatch = packets_[level.first + i].packet;
      if (dispatch.workgroup_size_x > 0 && dispatch.workgroup_size_y > 0 &&
          dispatch.workgroup_size_z > 0) {
        const std::uint64_t blocks =
            static_cast<std::uint64_t>(dispatch.grid_size_x / dispatch.workgroup_size_x) *
            (dispatch.grid_size_y / dispatch.workgroup_size_y) *
            (dispatch.grid_size_z / dispatch.workgroup_size_z);
        small = small && blocks <= runtime.scheduleMaxBlocks;
      }
    }
    double levelEnd = time + work;
    // the lane of each kernel: 0 the queue of the program, k > 0 helper k - 1
    std::vector<std::size_t> laneOf(level.count, 0);
    bool split = false;
    if (level.splittable && known && small && !helpers.empty() && work >= runtime.scheduleMinWork) {
      std::vector<std::size_t> order(level.count);
      for (std::size_t i = 0; i < level.count; ++i) {
        order[i] = i;
      }
      std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return estimates_[level.first + a] > estimates_[level.first + b];
      });
      std::vector<double> laneEnd(helpers.size() + 1);
      laneEnd[0] = time;
      for (std::size_t k = 0; k < helpers.size(); ++k) {
        laneEnd[k + 1] = std::max(time + runtime.forkSeconds, helperFree[k]);
      }
      for (const auto i : order) {
        std::size_t best = 0;
        for (std::size_t lane = 1; lane < laneEnd.size(); ++lane) {
          if (laneEnd[lane] < laneEnd[best]) {
            best = lane;
          }
        }
        laneOf[i] = best;
        laneEnd[best] += estimates_[level.first + i];
      }
      double end = laneEnd[0];
      bool any = false;
      for (std::size_t i = 0; i < level.count; ++i) {
        if (laneOf[i] > 0) {
          any = true;
          end = std::max(end, laneEnd[laneOf[i]] + runtime.forkSeconds);
        }
      }
      if (any && end < levelEnd) {
        split = true;
        levelEnd = end;
        for (std::size_t i = 0; i < level.count; ++i) {
          if (laneOf[i] > 0) {
            helperFree[laneOf[i] - 1] = laneEnd[laneOf[i]];
          }
        }
      }
    }
    if (!split) {
      for (std::size_t i = 0; i < level.count; ++i) {
        emit(queue_, level.first + i, false);
      }
      time = levelEnd;
      continue;
    }
    ++splitLevels;
    ensureLevelSignals(l, helpers.size());
    std::vector<bool> inLevel(helpers.size(), false);
    for (std::size_t i = 0; i < level.count; ++i) {
      if (laneOf[i] > 0) {
        inLevel[laneOf[i] - 1] = true;
        helperUsed[laneOf[i] - 1] = true;
        ++movedKernels;
      }
    }
    const bool logged = dagLog().on();
    if (logged) {
      dagLog().line("S %lld %p %d %zu %lld %lld %lld\n",
                    DagLog::now(),
                    static_cast<void*>(this),
                    queue_.number(),
                    l,
                    static_cast<long long>(time * 1e9),
                    static_cast<long long>(levelEnd * 1e9),
                    static_cast<long long>((time + work) * 1e9));
    }
    countDown(queue_, levelForks_[l]);
    const auto forkCount = ++levelForkCounts_[l];
    for (std::size_t k = 0; k < helpers.size(); ++k) {
      if (inLevel[k]) {
        const auto before = helperFree[k];
        waitCount(*helpers[k], levelForks_[l], forkCount);
        if (logged) {
          dagLog().line("F %d %llu %p %zu %lld\n",
                        helpers[k]->number(),
                        static_cast<unsigned long long>(helpers[k]->created()),
                        static_cast<void*>(this),
                        l,
                        static_cast<long long>(before * 1e9));
        }
      }
    }
    for (std::size_t i = 0; i < level.count; ++i) {
      emit(laneOf[i] == 0 ? queue_ : *helpers[laneOf[i] - 1], level.first + i, true);
    }
    for (std::size_t k = 0; k < helpers.size(); ++k) {
      if (inLevel[k]) {
        countDown(*helpers[k], levelJoins_[l][k]);
        waitCount(queue_, levelJoins_[l][k], ++levelJoinCounts_[l][k]);
        if (logged) {
          dagLog().line("J %d %llu %p %zu\n",
                        queue_.number(),
                        static_cast<unsigned long long>(queue_.created()),
                        static_cast<void*>(this),
                        l);
        }
      }
    }
    time = levelEnd;
  }
  queue_.predictUntil(time);
  for (std::size_t k = 0; k < helpers.size(); ++k) {
    if (helperUsed[k]) {
      helpers[k]->predictUntil(helperFree[k]);
    }
  }
  static const bool verbose = setting<bool>("VERBOSE", false);
  if (verbose) {
    static std::atomic<std::uint64_t> replays{0};
    static std::atomic<std::uint64_t> splits{0};
    static std::atomic<std::uint64_t> moved{0};
    splits += splitLevels;
    moved += movedKernels;
    if (++replays % 20000 == 0) {
      logInfo() << "HSA queues: in" << replays.load() << "replays, graphs spread" << splits.load()
                << "levels, with" << moved.load() << "kernels on other queues";
    }
  }
  lastIndex_ = queue_.created();
  return lastIndex_;
}

} // namespace device::hsaqueues
