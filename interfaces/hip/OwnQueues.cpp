// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "OwnQueues.h"

#include "Internals.h"
#include "utils/env.h"
#include "utils/logger.h"

#include <algorithm>
#include <amd_comgr/amd_comgr.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <deque>
#include <hip/hip_runtime.h>
#include <hsa/amd_hsa_queue.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace device::ownqueues {
namespace {

// ------------------------------------------------------------------------------------------------
// kernels of our own: copies, fills, and waits where the command processor cannot wait

__global__ void ownCopy(char* dst, const char* src, std::size_t bytes) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (((reinterpret_cast<std::uintptr_t>(dst) | reinterpret_cast<std::uintptr_t>(src)) & 15U) ==
      0) {
    const std::size_t vectors = bytes / 16;
    auto* dstVectors = reinterpret_cast<uint4*>(dst);
    const auto* srcVectors = reinterpret_cast<const uint4*>(src);
    for (std::size_t i = start; i < vectors; i += stride) {
      dstVectors[i] = srcVectors[i];
    }
    for (std::size_t i = vectors * 16 + start; i < bytes; i += stride) {
      dst[i] = src[i];
    }
  } else {
    for (std::size_t i = start; i < bytes; i += stride) {
      dst[i] = src[i];
    }
  }
}

// `width` elements of `elementSize` bytes (1, 2 or 4) in each of `height` rows, `pitch` bytes apart
__global__ void ownFill(char* dst,
                        std::uint32_t value,
                        std::uint32_t elementSize,
                        std::size_t width,
                        std::size_t height,
                        std::size_t pitch) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (std::size_t i = start; i < width * height; i += stride) {
    char* element = dst + (i / width) * pitch + (i % width) * elementSize;
    if (elementSize == 4) {
      *reinterpret_cast<std::uint32_t*>(element) = value;
    } else if (elementSize == 2) {
      *reinterpret_cast<std::uint16_t*>(element) = static_cast<std::uint16_t>(value);
    } else {
      *element = static_cast<char>(value);
    }
  }
}

// until the value of a signal is below `below`
__global__ void ownWaitBelow(const volatile std::int64_t* value, std::int64_t below) {
  while (*value >= below) {
    __builtin_amdgcn_s_sleep(2);
  }
}

// until the value of a signal is at least `atLeast`
__global__ void ownWaitAtLeast(const volatile std::int64_t* value, std::int64_t atLeast) {
  while (*value < atLeast) {
    __builtin_amdgcn_s_sleep(2);
  }
}

// until a counter in host memory is at least `atLeast`
// takes the time (by its completion signal, on a queue with profiling)
__global__ void ownStamp() {}

// keeps a compute unit busy for `ticks` of the constant clock (see samePipe())
__global__ void ownSpin(std::uint64_t ticks) {
  const auto start = wall_clock64();
  while (static_cast<std::uint64_t>(wall_clock64() - start) < ticks) {
    __builtin_amdgcn_s_sleep(1);
  }
}

__global__ void ownWaitHost(const volatile std::uint32_t* counter, std::uint32_t atLeast) {
  while (*counter < atLeast) {
    __builtin_amdgcn_s_sleep(2);
  }
}

// on a stream of HIP, which cannot wait for a signal of ours: until the counter of the marks of an
// own queue is below `below` (loads of the system read past the L2 caches, which the command
// processor does not update)
__global__ void ownHipWaitBelow(const std::int64_t* value, std::int64_t below) {
  while (__hip_atomic_load(value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) >= below) {
    __builtin_amdgcn_s_sleep(2);
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

// on a stream of HIP, after the work before it there: sets a counter that own queues wait for (see
// recordEvent())
__global__ void ownWriteCounter(std::uint32_t* counter, std::uint32_t value) {
  __hip_atomic_store(counter, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// lets the work after it see what the host and other agents wrote before, together with the
// acquire fence of the agent of the next kernel (for the caches of the other compute units): a
// fence of the system in the shader drops only the lines of the L2 cache that may be stale (on
// gfx90a, those of memory that is not local; the hardware keeps those of local memory coherent),
// while that of a packet drops all of them, at the expense of the kernels of all queues (on the
// MI250X, a kernel that read 4 MiB again and again took 3.3 times as long next to a queue of such
// packets, and 1.1 times next to one of such kernels)
__global__ void ownAcquireSystem() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""); }

// ------------------------------------------------------------------------------------------------

// the marks of a queue count its counter down from here; only one queue may count a signal down
// (the command processors of gfx90a lose decrements of a signal that several queues count down)
constexpr std::int64_t CounterStart = std::int64_t{1} << 62;
// the most packets an own queue holds, and the default memory for the arguments of its kernels: the
// host must never wait for space while the queue waits for work the host has yet to enqueue (e.g.
// for halo data that MPI receives on the host), so both should hold the work of an epoch
constexpr std::uint32_t MaxQueuePackets = 1U << 17;
constexpr std::uint64_t DefaultKernargRingMiB = 128;
constexpr std::uint32_t CopyThreads = 256;
// the long kernel of samePipe()
constexpr double SpinMicroseconds = 300;
// what the predicted work of a queue counts for a kernel of unknown duration, a copy, and for a
// wait for the host (see Program::submitScheduled())
constexpr double DefaultKernelSeconds = 5e-6;
constexpr double DefaultCopySeconds = 5e-6;
constexpr double DefaultHostSeconds = 50e-6;
constexpr std::size_t MaxMarkPredictions = 4096;

double steadySeconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// more pipes than this: the queues do not share them (see Runtime::placeQueue()); queues that get
// created but stay unused, at most (a GCD of the MI250X has 24 hardware queues for the processes)
constexpr std::size_t MaxPipes = 8;
constexpr std::size_t MaxSpareQueues = 8;
// the numbers of the helper queues of graphs start here (see Program)
constexpr int FirstHelperQueue = 1000000;
constexpr std::uint32_t MaxCopyBlocks = 1024;

constexpr std::uint16_t fenceBits(hsa_fence_scope_t acquire, hsa_fence_scope_t release) {
  return static_cast<std::uint16_t>((acquire << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                                    (release << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
}

constexpr std::uint16_t KernelFences = fenceBits(HSA_FENCE_SCOPE_AGENT, HSA_FENCE_SCOPE_AGENT);
// a mark makes all work before it visible to the host and other agents
constexpr std::uint16_t MarkFences = fenceBits(HSA_FENCE_SCOPE_NONE, HSA_FENCE_SCOPE_SYSTEM);
// a wait for the host (or work that it stands for, e.g. data from MPI)
constexpr std::uint16_t WaitFences = fenceBits(HSA_FENCE_SCOPE_SYSTEM, HSA_FENCE_SCOPE_NONE);
// a wait for another own queue, i.e. for work of the same device: the scope of the device is
// enough, while one of the system invalidates its L2 cache, at the expense of the kernels of all
// queues (actor-mpi of SeisSol on the MI250X: a fifth of its time)
constexpr std::uint16_t QueueWaitFences = fenceBits(HSA_FENCE_SCOPE_AGENT, HSA_FENCE_SCOPE_NONE);
constexpr std::uint16_t BarrierBit = 1U << HSA_PACKET_HEADER_BARRIER;

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

void check(hsa_status_t status, const char* call) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* text = nullptr;
    hsa_status_string(status, &text);
    logError() << "Own queues: HSA failed in" << call << "with" << (text != nullptr ? text : "?");
  }
}

void checkComgr(amd_comgr_status_t status, const char* call) {
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    logError() << "Own queues: comgr failed in" << call << "with" << static_cast<int>(status);
  }
}

void warnOnce(const std::string& message) {
  static std::mutex mutex;
  static std::unordered_set<std::string> warned;
  const std::lock_guard lock(mutex);
  if (warned.insert(message).second) {
    logWarning() << "Own queues:" << message;
  }
}

// ------------------------------------------------------------------------------------------------
// kernels as the loader and the metadata of their code objects know them

enum class ArgKind : std::uint8_t {
  Explicit,
  BlockCountX,
  BlockCountY,
  BlockCountZ,
  GroupSizeX,
  GroupSizeY,
  GroupSizeZ,
  GridDims,
  DynamicLdsSize,
  QueuePtr,
  PrivateBase,
  SharedBase,
  Zero,
  // features of HIP that we do not provide (printf, hostcall, heap, multi-grid, device enqueue)
  Unsupported
};

struct ArgInfo {
  std::uint32_t offset{0};
  std::uint32_t size{0};
  ArgKind kind{ArgKind::Zero};
};

struct KernelInfo {
  std::string name;
  std::uint64_t object{0};
  std::uint32_t kernargSize{0};
  std::uint32_t kernargAlign{16};
  std::uint32_t groupSize{0};
  std::uint32_t privateSize{0};
  bool dynamicStack{false};
  bool metadata{false};
  bool unsupported{false};
  std::vector<ArgInfo> args;
};

ArgKind argKind(const std::string& kind) {
  static const std::unordered_map<std::string, ArgKind> kinds{
      {"hidden_block_count_x", ArgKind::BlockCountX},
      {"hidden_block_count_y", ArgKind::BlockCountY},
      {"hidden_block_count_z", ArgKind::BlockCountZ},
      {"hidden_group_size_x", ArgKind::GroupSizeX},
      {"hidden_group_size_y", ArgKind::GroupSizeY},
      {"hidden_group_size_z", ArgKind::GroupSizeZ},
      {"hidden_grid_dims", ArgKind::GridDims},
      {"hidden_dynamic_lds_size", ArgKind::DynamicLdsSize},
      {"hidden_queue_ptr", ArgKind::QueuePtr},
      {"hidden_private_base", ArgKind::PrivateBase},
      {"hidden_shared_base", ArgKind::SharedBase},
      {"hidden_remainder_x", ArgKind::Zero},
      {"hidden_remainder_y", ArgKind::Zero},
      {"hidden_remainder_z", ArgKind::Zero},
      {"hidden_global_offset_x", ArgKind::Zero},
      {"hidden_global_offset_y", ArgKind::Zero},
      {"hidden_global_offset_z", ArgKind::Zero},
      {"hidden_none", ArgKind::Zero},
      {"hidden_printf_buffer", ArgKind::Unsupported},
      {"hidden_hostcall_buffer", ArgKind::Unsupported},
      {"hidden_heap_v1", ArgKind::Unsupported},
      {"hidden_multigrid_sync_arg", ArgKind::Unsupported},
      {"hidden_default_queue", ArgKind::Unsupported},
      {"hidden_completion_action", ArgKind::Unsupported},
  };
  if (kind.rfind("hidden_", 0) != 0) {
    return ArgKind::Explicit;
  }
  const auto found = kinds.find(kind);
  if (found == kinds.end()) {
    warnOnce("unknown hidden kernel argument " + kind + " left at 0");
    return ArgKind::Zero;
  }
  return found->second;
}

std::string metadataString(amd_comgr_metadata_node_t node) {
  std::size_t size = 0;
  checkComgr(amd_comgr_get_metadata_string(node, &size, nullptr), "get_metadata_string");
  std::string text(size, '\0');
  checkComgr(amd_comgr_get_metadata_string(node, &size, text.data()), "get_metadata_string");
  while (!text.empty() && text.back() == '\0') {
    text.pop_back();
  }
  return text;
}

bool lookupString(amd_comgr_metadata_node_t map, const char* key, std::string& value) {
  amd_comgr_metadata_node_t node;
  if (amd_comgr_metadata_lookup(map, key, &node) != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }
  amd_comgr_metadata_kind_t kind{};
  checkComgr(amd_comgr_get_metadata_kind(node, &kind), "get_metadata_kind");
  const bool found = kind == AMD_COMGR_METADATA_KIND_STRING;
  if (found) {
    value = metadataString(node);
  }
  amd_comgr_destroy_metadata(node);
  return found;
}

std::uint32_t lookupNumber(amd_comgr_metadata_node_t map, const char* key, std::uint32_t fallback) {
  std::string value;
  return lookupString(map, key, value) ? static_cast<std::uint32_t>(std::stoul(value)) : fallback;
}

// ------------------------------------------------------------------------------------------------

class Queue;

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

class Runtime {
  public:
  static Runtime& instance() {
    static Runtime runtime;
    return runtime;
  }

  bool setUp();
  const KernelInfo& kernel(const void* function);
  Queue& queue(int number);
  std::vector<Queue*> allQueues();
  void finalize();

  /// the short name of a kernel object, for the trace
  std::string kernelName(std::uint64_t object);
  /// the number of the queue whose marks count the signal down, or nothing
  std::optional<int> queueOfCounter(std::uint64_t signal);

  // the trace (see trace()): whether the queues give their packets signals that take the times,
  // from a pool
  std::atomic<bool> tracing{false};
  hsa_signal_t traceSignal();
  void freeTraceSignal(hsa_signal_t signal);

  hsa_agent_t gpu{};
  hsa_agent_t cpu{};
  hsa_amd_memory_pool_t kernargPool{};
  bool havePool{false};
  // kernel arguments in device memory that the host writes (DEVICE_OWN_QUEUE_KERNARGS=device),
  // instead of host memory
  hsa_amd_memory_pool_t deviceKernargPool{};
  bool haveDevicePool{false};
  bool deviceKernargs{false};
  // the fences of marks and of waits for other queues (DEVICE_OWN_QUEUE_MARK_SCOPE, system by
  // default, and DEVICE_OWN_QUEUE_WAIT_SCOPE, agent by default: system or agent)
  std::uint16_t markFences{MarkFences};
  std::uint16_t waitFences{QueueWaitFences};
  // how the work after a wait for the host, and copies from host memory, see what it wrote: by a
  // kernel that fences in the shader (DEVICE_OWN_QUEUE_HOST_ACQUIRE=kernel, the default, see
  // ownAcquireSystem), or by the acquire fence of the packet (=packet)
  bool acquireByKernel{true};
  // events time what they enclose (DEVICE_OWN_QUEUE_TIMING=1): by a kernel that takes the time
  bool timing{false};
  // waits between own streams and streams of HIP (e.g. those of NCCL) on the device
  // (DEVICE_OWN_QUEUE_HIP_WAITS=device, the default; =host waits on the host): a stream of HIP
  // waits for a mark by a kernel of ours (ownHipWaitBelow), an own stream for an event of HIP by a
  // counter that a kernel of ours sets on that stream after each event (see recordEvent())
  bool hipWaitsOnDevice{true};
  // DEVICE_OWN_QUEUE_SCHEDULE=1: graphs spread the kernels of their levels over own queues on other
  // pipes where that ends a level earlier, by the durations they learn and the predicted work of
  // the queues (see Program::submitScheduled()); a level gets spread only with at least
  // DEVICE_OWN_QUEUE_SCHEDULE_MIN_US of work, with forks and joins that take ..._FORK_US each
  bool schedule{false};
  double scheduleMinWork{120e-6};
  double forkSeconds{5e-6};
  // a level gets spread only if none of its kernels has more workgroups than this
  // (DEVICE_OWN_QUEUE_SCHEDULE_MAX_BLOCKS, the compute units of the device by default): larger ones
  // fill the device by themselves, and spreading them made a mesh of 1.4 million cells slower
  std::uint64_t scheduleMaxBlocks{128};
  /// the duration of a kernel (by its kernel object) the graphs have measured, or 0
  double kernelEstimate(std::uint64_t object);
  void learnKernel(std::uint64_t object, double seconds);
  // DEVICE_OWN_QUEUE_PIPES=check: each new queue gets checked for the pipes it shares; =spread: own
  // queues get pipes of their own as long as there are (see placeQueue())
  bool pipeCheck{false};
  bool pipeSpread{false};
  // own queues that are idle most of the time, which may share a pipe with another one
  // (DEVICE_OWN_QUEUE_LIGHT, a list of queue numbers)
  std::set<int> lightQueues;
  /**
   * An HSA queue for the own queue `number`, on a pipe with as few other own queues as possible.
   * The hardware scheduler gives the i-th queue of a process the pipe i modulo their number (the
   * MI250X: 4 pipes for the processes): queues that get created later take the pipes in turn, and
   * those after a queue that goes move up by one. So the first call finds the pipes, with an idle
   * queue on each (until a new queue is on a known one), and each call creates queues until one is
   * on a pipe as little busy as any (by its own queues that are not light, see lightQueues); the
   * others stay unused, as do the idle ones: destroyed, they would move the queues after them.
   */
  hsa_queue_t* placeQueue(int number);
  /// the own queues that the graphs of `self` may lend lanes from: on other pipes (if known), alone
  /// there but for light queues, not light themselves, no helpers
  std::vector<Queue*> borrowable(const Queue& self);
  std::uint64_t timestampFrequency{1};
  // the system timestamp when the own queues started; stamps before it are not valid
  std::uint64_t startTimestamp{0};
  hsa_ven_amd_loader_1_03_pfn_t loader{};
  // whether the command processor waits for signal values itself (barrier-value packets)
  bool barrierValue{false};
  std::uint32_t stackSize{1024};
  std::uint32_t queuePackets{1U << 14};
  std::uint64_t kernargRingBytes{DefaultKernargRingMiB << 20};
  std::uint32_t bus{0};
  std::uint32_t slot{0};
  std::uint32_t domain{0};
  // the compute units that the waves of a queue may use, by queue number
  // (DEVICE_OWN_QUEUE_CU_MASKS, e.g. "-10:0-23;1:24-109"); all of them for the others
  std::map<int, std::vector<std::uint32_t>> computeUnitMasks;
  std::uint32_t computeUnits{0};
  HostThread hostThread;

  // the symbols of the kernels, from the loaded code objects; by their names (with .kd)
  std::unordered_map<std::string, KernelInfo> symbols;
  std::unordered_set<std::uint64_t> parsedCodeObjects;

  private:
  void scanCodeObjects();

  std::mutex mutex_;
  std::unordered_map<const void*, const KernelInfo*> kernels_;
  std::map<int, std::unique_ptr<Queue>> queues_;
  // the pipes found so far (see placeQueue()): an idle queue on each, and the own queues on it
  struct Pipe {
    hsa_queue_t* probe{nullptr};
    std::vector<int> queues;
  };
  std::mutex estimateMutex_;
  std::unordered_map<std::uint64_t, double> kernelEstimates_;
  std::mutex placeMutex_;
  std::vector<Pipe> pipes_;
  bool pipesKnown_{false};
  // queues that stay unused: destroyed, they could move the other queues to other pipes
  std::vector<hsa_queue_t*> spareQueues_;
  std::unordered_map<std::uint64_t, std::string> kernelNames_;
  std::mutex traceMutex_;
  std::vector<hsa_signal_t> traceSignals_;
};

hsa_status_t visitAgent(hsa_agent_t agent, void* data) {
  auto& runtime = *static_cast<Runtime*>(data);
  hsa_device_type_t type{};
  hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (type == HSA_DEVICE_TYPE_CPU && runtime.cpu.handle == 0) {
    runtime.cpu = agent;
  }
  if (type == HSA_DEVICE_TYPE_GPU) {
    std::uint32_t location = 0;
    std::uint32_t domain = 0;
    hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &location);
    hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN), &domain);
    if (((location >> 8U) & 0xffU) == runtime.bus && ((location >> 3U) & 0x1fU) == runtime.slot &&
        domain == runtime.domain) {
      runtime.gpu = agent;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t visitPool(hsa_amd_memory_pool_t pool, void* data) {
  auto& runtime = *static_cast<Runtime*>(data);
  std::uint32_t flags = 0;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0 && !runtime.havePool) {
    runtime.kernargPool = pool;
    runtime.havePool = true;
  }
  return HSA_STATUS_SUCCESS;
}

// fine-grained device memory, which the host can write without the device caching stale copies
hsa_status_t visitDevicePool(hsa_amd_memory_pool_t pool, void* data) {
  auto& runtime = *static_cast<Runtime*>(data);
  hsa_amd_segment_t segment{};
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  std::uint32_t flags = 0;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  bool allowed = false;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allowed);
  if (segment == HSA_AMD_SEGMENT_GLOBAL && allowed &&
      (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) != 0 && !runtime.haveDevicePool) {
    runtime.deviceKernargPool = pool;
    runtime.haveDevicePool = true;
  }
  return HSA_STATUS_SUCCESS;
}

/// memory for kernel arguments: in device memory that the host writes, or in host memory
void* allocKernargMemory(std::size_t bytes) {
  auto& runtime = Runtime::instance();
  void* memory = nullptr;
  if (runtime.deviceKernargs) {
    check(hsa_amd_memory_pool_allocate(runtime.deviceKernargPool, bytes, 0, &memory),
          "hsa_amd_memory_pool_allocate");
    const hsa_agent_t agents[] = {runtime.gpu, runtime.cpu};
    check(hsa_amd_agents_allow_access(2, agents, nullptr, memory), "hsa_amd_agents_allow_access");
  } else {
    check(hsa_amd_memory_pool_allocate(runtime.kernargPool, bytes, 0, &memory),
          "hsa_amd_memory_pool_allocate");
    check(hsa_amd_agents_allow_access(1, &runtime.gpu, nullptr, memory),
          "hsa_amd_agents_allow_access");
  }
  return memory;
}

/// after the host has written kernel arguments into device memory: they are there before the
/// packet that reads them gets published
void flushKernargs(const void* last) {
  if (Runtime::instance().deviceKernargs) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    static_cast<void>(*static_cast<const volatile std::uint8_t*>(last));
  }
}

void readMetadata(Runtime& runtime, const void* elf, std::size_t size) {
  amd_comgr_data_t data;
  checkComgr(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data), "create_data");
  checkComgr(amd_comgr_set_data(data, size, static_cast<const char*>(elf)), "set_data");
  amd_comgr_metadata_node_t root;
  if (amd_comgr_get_data_metadata(data, &root) != AMD_COMGR_STATUS_SUCCESS) {
    amd_comgr_release_data(data);
    return;
  }
  amd_comgr_metadata_node_t kernels;
  if (amd_comgr_metadata_lookup(root, "amdhsa.kernels", &kernels) == AMD_COMGR_STATUS_SUCCESS) {
    std::size_t count = 0;
    checkComgr(amd_comgr_get_metadata_list_size(kernels, &count), "get_metadata_list_size");
    for (std::size_t k = 0; k < count; ++k) {
      amd_comgr_metadata_node_t kernel;
      checkComgr(amd_comgr_index_list_metadata(kernels, k, &kernel), "index_list_metadata");
      std::string symbol;
      lookupString(kernel, ".symbol", symbol);
      auto& info = runtime.symbols[symbol];
      lookupString(kernel, ".name", info.name);
      info.metadata = true;
      info.kernargSize = lookupNumber(kernel, ".kernarg_segment_size", info.kernargSize);
      info.kernargAlign = std::max<std::uint32_t>(
          16, lookupNumber(kernel, ".kernarg_segment_align", info.kernargAlign));
      info.groupSize = lookupNumber(kernel, ".group_segment_fixed_size", info.groupSize);
      info.privateSize = lookupNumber(kernel, ".private_segment_fixed_size", info.privateSize);
      std::string stack;
      info.dynamicStack =
          lookupString(kernel, ".uses_dynamic_stack", stack) && (stack == "1" || stack == "true");
      info.args.clear();
      info.unsupported = false;
      amd_comgr_metadata_node_t args;
      if (amd_comgr_metadata_lookup(kernel, ".args", &args) == AMD_COMGR_STATUS_SUCCESS) {
        std::size_t argCount = 0;
        checkComgr(amd_comgr_get_metadata_list_size(args, &argCount), "get_metadata_list_size");
        for (std::size_t a = 0; a < argCount; ++a) {
          amd_comgr_metadata_node_t arg;
          checkComgr(amd_comgr_index_list_metadata(args, a, &arg), "index_list_metadata");
          std::string kind;
          lookupString(arg, ".value_kind", kind);
          ArgInfo argInfo;
          argInfo.kind = argKind(kind);
          argInfo.offset = lookupNumber(arg, ".offset", 0);
          argInfo.size = lookupNumber(arg, ".size", 0);
          info.unsupported = info.unsupported || argInfo.kind == ArgKind::Unsupported;
          info.args.push_back(argInfo);
          amd_comgr_destroy_metadata(arg);
        }
        amd_comgr_destroy_metadata(args);
      }
      amd_comgr_destroy_metadata(kernel);
    }
    amd_comgr_destroy_metadata(kernels);
  }
  amd_comgr_destroy_metadata(root);
  amd_comgr_release_data(data);
}

hsa_status_t
    visitCodeObject(hsa_executable_t /*executable*/, hsa_loaded_code_object_t object, void* data) {
  auto& runtime = *static_cast<Runtime*>(data);
  hsa_agent_t agent{};
  runtime.loader.hsa_ven_amd_loader_loaded_code_object_get_info(
      object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT, &agent);
  if (agent.handle != runtime.gpu.handle) {
    return HSA_STATUS_SUCCESS;
  }
  std::uint32_t storage = 0;
  runtime.loader.hsa_ven_amd_loader_loaded_code_object_get_info(
      object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE, &storage);
  if (storage != HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
    return HSA_STATUS_SUCCESS;
  }
  std::uint64_t base = 0;
  std::uint64_t size = 0;
  runtime.loader.hsa_ven_amd_loader_loaded_code_object_get_info(
      object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE, &base);
  runtime.loader.hsa_ven_amd_loader_loaded_code_object_get_info(
      object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE, &size);
  if (base != 0 && runtime.parsedCodeObjects.insert(base).second) {
    readMetadata(runtime, reinterpret_cast<const void*>(base), size);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t visitSymbol(hsa_executable_t /*executable*/,
                         hsa_agent_t /*agent*/,
                         hsa_executable_symbol_t symbol,
                         void* data) {
  auto& runtime = *static_cast<Runtime*>(data);
  hsa_symbol_kind_t kind{};
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind);
  if (kind != HSA_SYMBOL_KIND_KERNEL) {
    return HSA_STATUS_SUCCESS;
  }
  std::uint32_t length = 0;
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &length);
  std::string name(length, '\0');
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data());
  auto& info = runtime.symbols[name];
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &info.object);
  if (!info.metadata) {
    hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &info.kernargSize);
    hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &info.groupSize);
    hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &info.privateSize);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t visitExecutable(hsa_executable_t executable, void* data) {
  auto& runtime = *static_cast<Runtime*>(data);
  runtime.loader.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
      executable, visitCodeObject, data);
  hsa_executable_iterate_agent_symbols(executable, runtime.gpu, visitSymbol, data);
  return HSA_STATUS_SUCCESS;
}

void queueError(hsa_status_t status, hsa_queue_t* queue, void* /*data*/) {
  const char* text = nullptr;
  hsa_status_string(status, &text);
  logError() << "Own queues: the queue" << static_cast<const void*>(queue) << "failed with"
             << (text != nullptr ? text : "?");
}

// ------------------------------------------------------------------------------------------------

/// a packet as it goes into a queue: all of its 64 bytes, the header word included
struct PacketImage {
  std::array<std::uint8_t, 64> bytes{};

  template <typename Packet>
  Packet& as() {
    static_assert(sizeof(Packet) == 64);
    return *reinterpret_cast<Packet*>(bytes.data());
  }

  void setHeader(std::uint16_t header, std::uint16_t rest) {
    const auto word =
        static_cast<std::uint32_t>(header) | (static_cast<std::uint32_t>(rest) << 16U);
    std::memcpy(bytes.data(), &word, sizeof(word));
  }
};

hsa_queue_t* createHsaQueue() {
  auto& runtime = Runtime::instance();
  hsa_queue_t* queue = nullptr;
  check(hsa_queue_create(runtime.gpu,
                         runtime.queuePackets,
                         HSA_QUEUE_TYPE_MULTI,
                         queueError,
                         nullptr,
                         UINT32_MAX,
                         UINT32_MAX,
                         &queue),
        "hsa_queue_create");
  return queue;
}

/// a kernel of one wave on a queue that nothing else writes to (see Runtime::placeQueue())
void dispatchAlone(hsa_queue_t* queue,
                   const KernelInfo& kernel,
                   const void* kernargs,
                   hsa_signal_t completion) {
  PacketImage image;
  auto& packet = image.as<hsa_kernel_dispatch_packet_t>();
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 64;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.private_segment_size = kernel.privateSize;
  packet.group_segment_size = kernel.groupSize;
  packet.kernel_object = kernel.object;
  packet.kernarg_address = const_cast<void*>(kernargs);
  packet.completion_signal = completion;
  image.setHeader(
      static_cast<std::uint16_t>((HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                                 BarrierBit | KernelFences),
      static_cast<std::uint16_t>(3U << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS));
  const auto index = hsa_queue_add_write_index_relaxed(queue, 1);
  auto* slot = static_cast<std::uint8_t*>(queue->base_address) + (index % queue->size) * 64;
  std::memcpy(slot + sizeof(std::uint32_t),
              image.bytes.data() + sizeof(std::uint32_t),
              image.bytes.size() - sizeof(std::uint32_t));
  std::uint32_t word = 0;
  std::memcpy(&word, image.bytes.data(), sizeof(word));
  __atomic_store_n(reinterpret_cast<std::uint32_t*>(slot), word, __ATOMIC_RELEASE);
  hsa_signal_store_screlease(queue->doorbell_signal, static_cast<hsa_signal_value_t>(index));
}

/**
 * Whether two queues that nothing else uses share a pipe of the command processor. The hardware
 * scheduler gives the queues of a process to its pipes one after the other, and a pipe serves its
 * queues together (MI250X, ROCm 6.3): a kernel that one queue starts after a long one of another
 * queue on the pipe completes only after it, and a queue that waits in a barrier packet holds up
 * the others on the pipe for about 40 us at a time. Here: a long kernel on `first`, then a short
 * one on `second`.
 */
bool samePipe(hsa_queue_t* first, hsa_queue_t* second) {
  auto& runtime = Runtime::instance();
  const auto& spin = runtime.kernel(reinterpret_cast<const void*>(ownSpin));
  const auto& stamp = runtime.kernel(reinterpret_cast<const void*>(ownStamp));
  // (the hidden arguments stay zero: the kernels do not read them)
  static const auto* spinArgs = [&]() {
    int rate = 0;
    APIWRAP(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0));
    const auto bytes = std::max<std::uint32_t>(spin.kernargSize, 64);
    auto* kernargs = static_cast<std::uint8_t*>(allocKernargMemory(bytes));
    std::memset(kernargs, 0, bytes);
    const auto ticks = static_cast<std::uint64_t>(SpinMicroseconds * rate / 1000.0);
    std::memcpy(kernargs, &ticks, sizeof(ticks));
    flushKernargs(kernargs + bytes - 1);
    return kernargs;
  }();
  static const auto* noArgs = [&]() {
    const auto bytes = std::max<std::uint32_t>(stamp.kernargSize, 64);
    auto* kernargs = static_cast<std::uint8_t*>(allocKernargMemory(bytes));
    std::memset(kernargs, 0, bytes);
    flushKernargs(kernargs + bytes - 1);
    return kernargs;
  }();
  hsa_signal_t spun{};
  hsa_signal_t probed{};
  check(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &spun),
        "hsa_amd_signal_create");
  check(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &probed),
        "hsa_amd_signal_create");
  dispatchAlone(first, spin, spinArgs, spun);
  // (the long kernel has started by then)
  const auto settle = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - settle < std::chrono::microseconds(30)) {
  }
  const auto start = std::chrono::steady_clock::now();
  dispatchAlone(second, stamp, noArgs, probed);
  while (hsa_signal_wait_scacquire(
             probed, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) > 0) {
  }
  const auto took =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
  while (hsa_signal_wait_scacquire(
             spun, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) > 0) {
  }
  hsa_signal_destroy(spun);
  hsa_signal_destroy(probed);
  return took > SpinMicroseconds / 2;
}

/**
 * An HSA queue of our own. Its marks (barrier packets that wait for all work before them) count its
 * counter signal down, from CounterStart; mark m has been reached once the counter is at most
 * CounterStart - m. The arguments of the kernels come from a ring, which marks free again.
 *
 * The host never waits to enqueue: the queue may wait for work that the host has yet to enqueue
 * (e.g. for halo data that MPI receives on the host). Packets that find the queue full wait in host
 * memory until the host thread (or the next packet) writes them into it; arguments that find the
 * ring full get memory of their own, which the next mark frees again.
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
    if (const auto mask = runtime.computeUnitMasks.find(number);
        mask != runtime.computeUnitMasks.end()) {
      check(hsa_amd_queue_cu_set_mask(
                queue_, static_cast<std::uint32_t>(mask->second.size() * 32), mask->second.data()),
            "hsa_amd_queue_cu_set_mask");
    }
    if (runtime.timing || runtime.tracing.load() || runtime.schedule) {
      // before the first packet: enabled later, the first stamps of the queue had no times
      enableProfiling();
    }
    ringBytes_ = runtime.kernargRingBytes;
    ring_ = static_cast<char*>(allocKernargMemory(ringBytes_));
    APIWRAP(hipStreamCreateWithFlags(&helper_, hipStreamNonBlocking));
    if (utils::Env("DEVICE_").get<bool>("OWN_QUEUE_VERBOSE", false)) {
      logInfo() << "Own queue" << number << ": packets at" << queue_->base_address
                << ", marks count" << reinterpret_cast<void*>(counter_.handle);
    }
  }

  ~Queue() {
    // (the host thread has written all packets by now, see finalize())
    const auto last = markCoveringLocked(created_);
    wait(last);
    reclaim();
    for (auto* arena : arenas_) {
      static_cast<void>(hipFree(arena));
    }
    static_cast<void>(hipStreamDestroy(helper_));
    hsa_amd_memory_pool_free(ring_);
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
    if (runtime_.tracing.load(std::memory_order_relaxed)) {
      PacketImage traced = packet;
      annotate(traced);
      pushImage(traced);
    } else {
      pushImage(packet);
    }
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
      warnOnce("an own queue is full, its packets wait in host memory (DEVICE_OWN_QUEUE_PACKETS)");
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
    warnOnce("the memory for kernel arguments of an own queue is used up; further arguments get "
             "memory of their own (DEVICE_OWN_QUEUE_KERNARG_MIB)");
    void* memory = allocKernargMemory(size);
    ownKernargs_.push_back(memory);
    return memory;
  }

  /**
   * Enqueues a mark: a barrier packet that completes once all work before it on the queue has, and
   * then makes it visible to the host and other agents. Returns its number.
   */
  std::uint64_t mark() {
    PacketImage packet;
    packet.as<hsa_barrier_and_packet_t>().completion_signal = counter_;
    packet.setHeader(
        static_cast<std::uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                   BarrierBit | runtime_.markFences),
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
    return overflow_.empty() && marks_ > 0 && lastMarkIndex_ == created_ && reached(marks_);
  }

  // the time (s of steady_clock) by which the queue is predicted to have done all it got, for
  // DEVICE_OWN_QUEUE_SCHEDULE (see Program::submitScheduled()): its work as learned or guessed, and
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

  /// lets the work enqueued from now on wait until the 32-bit counter in the signal reaches `value`
  void waitForCounter(hsa_signal_t signal, std::uint32_t value) {
    PacketImage packet;
    auto& barrier = packet.as<hsa_amd_barrier_value_packet_t>();
    barrier.signal = signal;
    barrier.value = value;
    barrier.mask = 0xffffffff;
    barrier.cond = HSA_SIGNAL_CONDITION_GTE;
    // (the host has written the data the counter stands for)
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
      dispatch(kernelOf(reinterpret_cast<const void*>(ownAcquireSystem)),
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
      dispatch(kernelOf(reinterpret_cast<const void*>(ownWaitBelow)),
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
               " may use printf, hostcalls, the device heap or device enqueue, which own queues do "
               "not provide");
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
    } else {
      volatile hsa_signal_value_t* gateValue = nullptr;
      check(hsa_amd_signal_value_pointer(gate, &gateValue), "hsa_amd_signal_value_pointer");
      const auto* value = const_cast<const std::int64_t*>(gateValue);
      std::int64_t atLeast = 1;
      void* args[] = {&value, &atLeast};
      dispatch(kernelOf(reinterpret_cast<const void*>(ownWaitAtLeast)),
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
    dispatch(kernelOf(reinterpret_cast<const void*>(ownStamp)),
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
      arenas_.push_back(arena);
      arenaBytes_ = bytes;
    }
    return arenas_.empty() ? nullptr : arenas_.back();
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
      for (auto* memory : ownReleases_.front().second) {
        hsa_amd_memory_pool_free(memory);
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
  std::vector<void*> ownKernargs_;
  std::deque<std::pair<std::uint64_t, std::vector<void*>>> ownReleases_;

  // for asynchronous allocations, which a stream of HIP that never gets any work makes
  hipStream_t helper_{nullptr};
  std::deque<std::pair<std::uint64_t, void*>> pendingFrees_;
  std::vector<void*> arenas_;
  std::size_t arenaBytes_{0};
};

bool Runtime::setUp() {
  int device = 0;
  APIWRAP(hipGetDevice(&device));
  int bus = 0;
  int slot = 0;
  int domain = 0;
  APIWRAP(hipDeviceGetAttribute(&bus, hipDeviceAttributePciBusId, device));
  APIWRAP(hipDeviceGetAttribute(&slot, hipDeviceAttributePciDeviceId, device));
  APIWRAP(hipDeviceGetAttribute(&domain, hipDeviceAttributePciDomainID, device));
  this->bus = static_cast<std::uint32_t>(bus);
  this->slot = static_cast<std::uint32_t>(slot);
  this->domain = static_cast<std::uint32_t>(domain);
  // (HIP has initialized HSA already; this only counts up)
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    return false;
  }
  check(hsa_iterate_agents(visitAgent, this), "hsa_iterate_agents");
  if (gpu.handle == 0 || cpu.handle == 0) {
    logWarning() << "Own queues: no HSA agent for the device.";
    return false;
  }
  check(hsa_amd_agent_iterate_memory_pools(cpu, visitPool, this),
        "hsa_amd_agent_iterate_memory_pools");
  if (!havePool) {
    logWarning() << "Own queues: no memory for kernel arguments.";
    return false;
  }
  if (hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(loader), &loader) !=
      HSA_STATUS_SUCCESS) {
    logWarning() << "Own queues: no loader extension.";
    return false;
  }
  char name[64] = {};
  hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, name);
  // the command processors of MI200 and MI300 wait for signal values (barrier-value packets; CLR
  // uses them only there), that of gfx1150 rejects them as malformed, the other ones are untried;
  // DEVICE_OWN_QUEUE_WAITS=kernel|packet decides otherwise
  const std::string arch(name);
  barrierValue = arch == "gfx90a" || arch.rfind("gfx94", 0) == 0 || arch == "gfx950";
  utils::Env env("DEVICE_");
  const auto waits = env.get<std::string>("OWN_QUEUE_WAITS", "");
  if (waits == "kernel") {
    barrierValue = false;
  } else if (waits == "packet") {
    barrierValue = true;
  }
  std::size_t stack = 0;
  if (hipDeviceGetLimit(&stack, hipLimitStackSize) == hipSuccess && stack > 0) {
    stackSize = static_cast<std::uint32_t>(stack);
  }
  std::uint32_t maxPackets = 0;
  hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &maxPackets);
  queuePackets =
      std::min(env.get<std::uint32_t>("OWN_QUEUE_PACKETS", std::uint32_t{MaxQueuePackets}),
               maxPackets > 0 ? maxPackets : MaxQueuePackets);
  kernargRingBytes =
      env.get<std::uint64_t>("OWN_QUEUE_KERNARG_MIB", std::uint64_t{DefaultKernargRingMiB}) << 20;
  timing = env.get<bool>("OWN_QUEUE_TIMING", false);
  hipWaitsOnDevice = env.get<std::string>("OWN_QUEUE_HIP_WAITS", "device") != "host";
  // the pipes of the command processor (see Runtime::placeQueue()): own queues spread over them on
  // the MI250X (measured there), DEVICE_OWN_QUEUE_PIPES=spread|check|any decides otherwise
  const auto pipes = env.get<std::string>("OWN_QUEUE_PIPES", arch == "gfx90a" ? "spread" : "any");
  pipeCheck = pipes == "check";
  pipeSpread = pipes == "spread";
  schedule = barrierValue && env.get<bool>("OWN_QUEUE_SCHEDULE", false);
  scheduleMinWork = env.get<double>("OWN_QUEUE_SCHEDULE_MIN_US", 120.0) * 1e-6;
  forkSeconds = env.get<double>("OWN_QUEUE_SCHEDULE_FORK_US", 5.0) * 1e-6;
  {
    const auto light = env.get<std::string>("OWN_QUEUE_LIGHT", "");
    std::size_t position = 0;
    while (position < light.size()) {
      const auto end = std::min(light.find(',', position), light.size());
      if (end > position) {
        lightQueues.insert(std::stoi(light.substr(position, end - position)));
      }
      position = end + 1;
    }
  }
  hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestampFrequency);
  if (timestampFrequency == 0) {
    timestampFrequency = 1;
  }
  hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &startTimestamp);
  // where the kernel arguments go: host memory (default), or device memory that the host writes
  if (env.get<std::string>("OWN_QUEUE_KERNARGS", "host") == "device") {
    check(hsa_amd_agent_iterate_memory_pools(gpu, visitDevicePool, this),
          "hsa_amd_agent_iterate_memory_pools");
    deviceKernargs = haveDevicePool;
    if (!haveDevicePool) {
      logWarning() << "Own queues: no fine-grained device memory for the kernel arguments; they "
                      "stay in host memory.";
    }
  }
  const auto scope = [&](const char* name, hsa_fence_scope_t fallback) {
    const auto value = env.get<std::string>(name, "");
    return value == "agent"    ? HSA_FENCE_SCOPE_AGENT
           : value == "system" ? HSA_FENCE_SCOPE_SYSTEM
                               : fallback;
  };
  markFences =
      fenceBits(HSA_FENCE_SCOPE_NONE, scope("OWN_QUEUE_MARK_SCOPE", HSA_FENCE_SCOPE_SYSTEM));
  waitFences =
      fenceBits(scope("OWN_QUEUE_WAIT_SCOPE", HSA_FENCE_SCOPE_AGENT), HSA_FENCE_SCOPE_NONE);
  acquireByKernel = env.get<std::string>("OWN_QUEUE_HOST_ACQUIRE", "kernel") != "packet";
  hsa_agent_get_info(
      gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT), &computeUnits);
  // (about a workgroup for each compute unit: kernels that wait for latencies more than for the
  // device; the MI250X has 110)
  scheduleMaxBlocks = env.get<std::uint64_t>("OWN_QUEUE_SCHEDULE_MAX_BLOCKS",
                                             computeUnits > 0 ? computeUnits : 128);
  {
    // "<queue>:<first>-<last>[+<first>-<last>...];..."
    const auto masks = env.get<std::string>("OWN_QUEUE_CU_MASKS", "");
    std::size_t position = 0;
    while (position < masks.size()) {
      const auto end = std::min(masks.find(';', position), masks.size());
      const auto entry = masks.substr(position, end - position);
      position = end + 1;
      const auto colon = entry.find(':');
      if (colon == std::string::npos) {
        continue;
      }
      std::vector<std::uint32_t> mask((computeUnits + 31) / 32, 0);
      std::size_t range = colon + 1;
      while (range < entry.size()) {
        const auto rangeEnd = std::min(entry.find('+', range), entry.size());
        const auto text = entry.substr(range, rangeEnd - range);
        range = rangeEnd + 1;
        const auto dash = text.find('-');
        const auto first = static_cast<std::uint32_t>(std::stoul(text.substr(0, dash)));
        const auto last = dash == std::string::npos
                              ? first
                              : static_cast<std::uint32_t>(std::stoul(text.substr(dash + 1)));
        for (auto unit = first; unit <= last && unit < computeUnits; ++unit) {
          mask[unit / 32] |= 1U << (unit % 32);
        }
      }
      computeUnitMasks[std::stoi(entry.substr(0, colon))] = mask;
    }
  }
  logInfo() << "Own queues on" << name << "; queues wait for other queues by"
            << (barrierValue ? "barrier-value packets" : "kernels") << "; work after the host sees"
            << "its data by" << (acquireByKernel ? "a fence in a kernel" : "packet fences")
            << "; kernel arguments in" << (deviceKernargs ? "device memory" : "host memory")
            << "; queues of" << queuePackets << "packets and" << (kernargRingBytes >> 20)
            << "MiB for kernel arguments";
  return true;
}

void Runtime::scanCodeObjects() {
  check(loader.hsa_ven_amd_loader_iterate_executables(visitExecutable, this),
        "hsa_ven_amd_loader_iterate_executables");
}

const KernelInfo& Runtime::kernel(const void* function) {
  const std::lock_guard lock(mutex_);
  const auto found = kernels_.find(function);
  if (found != kernels_.end()) {
    return *found->second;
  }
  // HIP loads the code object of a kernel when it is needed first
  hipFuncAttributes attributes{};
  static_cast<void>(hipFuncGetAttributes(&attributes, function));
  const char* name = hipKernelNameRefByPtr(function, nullptr);
  if (name == nullptr) {
    logError() << "Own queues: HIP does not know the kernel" << function;
  }
  const auto symbol = std::string(name) + ".kd";
  auto known = symbols.find(symbol);
  if (known == symbols.end() || known->second.object == 0) {
    scanCodeObjects();
    known = symbols.find(symbol);
  }
  if (known == symbols.end() || known->second.object == 0) {
    logError() << "Own queues: no kernel object for" << name;
  }
  if (!known->second.metadata) {
    logError() << "Own queues: no metadata for the kernel" << name
               << "(its code object was not loaded from memory)";
  }
  if (known->second.name.empty()) {
    known->second.name = name;
  }
  kernels_.emplace(function, &known->second);
  return known->second;
}

Queue& Runtime::queue(int number) {
  {
    const std::lock_guard lock(mutex_);
    const auto found = queues_.find(number);
    if (found != queues_.end()) {
      return *found->second;
    }
  }
  hsa_queue_t* placed = pipeCheck || pipeSpread ? placeQueue(number) : nullptr;
  const std::lock_guard lock(mutex_);
  auto& queue = queues_[number];
  if (queue != nullptr) {
    // (another thread was first)
    if (placed != nullptr) {
      const std::lock_guard placeLock(placeMutex_);
      spareQueues_.push_back(placed);
    }
    return *queue;
  }
  queue = std::make_unique<Queue>(*this, number, placed);
  return *queue;
}

double Runtime::kernelEstimate(std::uint64_t object) {
  const std::lock_guard lock(estimateMutex_);
  const auto found = kernelEstimates_.find(object);
  return found == kernelEstimates_.end() ? 0.0 : found->second;
}

void Runtime::learnKernel(std::uint64_t object, double seconds) {
  const std::lock_guard lock(estimateMutex_);
  auto& estimate = kernelEstimates_[object];
  estimate = estimate <= 0 ? seconds : 0.75 * estimate + 0.25 * seconds;
}

std::vector<Queue*> Runtime::borrowable(const Queue& self) {
  // the queues on the pipe of `self`, and those that share a pipe with another busy queue: work
  // lent to them would hold up the other queue on their pipe
  std::vector<int> excluded;
  {
    const std::lock_guard lock(placeMutex_);
    for (const auto& pipe : pipes_) {
      std::size_t busy = 0;
      for (const auto number : pipe.queues) {
        busy += lightQueues.count(number) > 0 || number >= FirstHelperQueue ? 0 : 1;
      }
      if (busy > 1 ||
          std::find(pipe.queues.begin(), pipe.queues.end(), self.number()) != pipe.queues.end()) {
        excluded.insert(excluded.end(), pipe.queues.begin(), pipe.queues.end());
      }
    }
  }
  std::vector<Queue*> candidates;
  const std::lock_guard lock(mutex_);
  for (const auto& [number, queue] : queues_) {
    if (queue.get() != &self && number < FirstHelperQueue && lightQueues.count(number) == 0 &&
        std::find(excluded.begin(), excluded.end(), number) == excluded.end()) {
      candidates.push_back(queue.get());
    }
  }
  return candidates;
}

hsa_queue_t* Runtime::placeQueue(int number) {
  const std::lock_guard lock(placeMutex_);
  const auto pipeOf = [&](hsa_queue_t* queue) -> int {
    for (std::size_t pipe = 0; pipe < pipes_.size(); ++pipe) {
      if (samePipe(queue, pipes_[pipe].probe)) {
        return static_cast<int>(pipe);
      }
    }
    return -1;
  };
  if (!pipesKnown_) {
    // an idle queue on each pipe: the queues of a process get the pipes one after the other, so
    // the first new queue on a known pipe means that there are no more
    while (true) {
      auto* queue = createHsaQueue();
      if (pipes_.size() < MaxPipes && pipeOf(queue) < 0) {
        pipes_.push_back({queue, {}});
        continue;
      }
      spareQueues_.push_back(queue);
      break;
    }
    pipesKnown_ = true;
    if (pipes_.size() >= MaxPipes) {
      // (queues that do not hold each other up)
      logWarning() << "Own queues: the queues do not seem to share pipes; they go anywhere";
      pipeSpread = false;
    } else {
      logInfo() << "Own queues: the command processor serves the queues of the process on"
                << pipes_.size() << "pipes";
    }
  }
  // how busy a pipe is: by its own queues that are not light, then by all of them (a light queue
  // goes rather to one with light queues only)
  const bool light = lightQueues.count(number) > 0;
  const auto load = [&](std::size_t pipe) {
    std::size_t heavy = 0;
    for (const auto other : pipes_[pipe].queues) {
      heavy += lightQueues.count(other) > 0 ? 0 : 1;
    }
    return light ? std::pair{pipes_[pipe].queues.size() - heavy, heavy}
                 : std::pair{heavy, pipes_[pipe].queues.size()};
  };
  hsa_queue_t* queue = nullptr;
  int pipe = -1;
  if (pipeSpread) {
    // a pipe as little busy as any (the next queues take the pipes in turn: a round of tries)
    auto least = load(0);
    for (std::size_t other = 1; other < pipes_.size(); ++other) {
      least = std::min(least, load(other));
    }
    for (std::size_t attempt = 0; attempt <= pipes_.size() && spareQueues_.size() < MaxSpareQueues;
         ++attempt) {
      queue = createHsaQueue();
      pipe = pipeOf(queue);
      if (pipe >= 0 && load(pipe) == least) {
        break;
      }
      spareQueues_.push_back(queue);
      queue = nullptr;
    }
    if (queue == nullptr) {
      logWarning() << "Own queue" << number << "found no pipe as little busy as" << least.first;
      queue = createHsaQueue();
      pipe = pipeOf(queue);
    }
  } else {
    queue = createHsaQueue();
    pipe = pipeOf(queue);
  }
  std::string others;
  if (pipe >= 0) {
    for (const auto other : pipes_[pipe].queues) {
      others += " " + std::to_string(other);
    }
    pipes_[pipe].queues.push_back(number);
  }
  if (pipeCheck || utils::Env("DEVICE_").get<bool>("OWN_QUEUE_VERBOSE", false)) {
    logInfo() << "Own queue" << number << "on pipe" << pipe << "with own queues"
              << (others.empty() ? std::string(" none") : others) << "(" << spareQueues_.size()
              << "spare queues so far)";
  }
  return queue;
}

std::vector<Queue*> Runtime::allQueues() {
  const std::lock_guard lock(mutex_);
  std::vector<Queue*> all;
  for (auto& [number, queue] : queues_) {
    all.push_back(queue.get());
  }
  return all;
}

std::string Runtime::kernelName(std::uint64_t object) {
  const std::lock_guard lock(mutex_);
  const auto known = kernelNames_.find(object);
  if (known != kernelNames_.end()) {
    return known->second;
  }
  std::string name = "kernel";
  for (const auto& [symbol, info] : symbols) {
    if (info.object == object) {
      const bool descriptor = symbol.size() > 3 && symbol.compare(symbol.size() - 3, 3, ".kd") == 0;
      name = symbol.substr(0, symbol.size() - (descriptor ? 3 : 0));
      break;
    }
  }
  int status = 0;
  char* demangled = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
  if (status == 0 && demangled != nullptr) {
    name = demangled;
  }
  std::free(demangled);
  // without the parameters and template arguments, and with at most two levels of names
  for (auto anonymous = name.find("(anonymous namespace)::"); anonymous != std::string::npos;
       anonymous = name.find("(anonymous namespace)::")) {
    name.erase(anonymous, std::string("(anonymous namespace)::").size());
  }
  std::string flat;
  int depth = 0;
  for (const char c : name.substr(0, name.find('('))) {
    depth += c == '<' ? 1 : c == '>' ? -1 : 0;
    if (depth == 0 && c != '>') {
      flat.push_back(c);
    }
  }
  const auto last = flat.rfind("::");
  if (last != std::string::npos && last > 0) {
    const auto before = flat.rfind("::", last - 1);
    if (before != std::string::npos) {
      flat = flat.substr(before + 2);
    }
  }
  return kernelNames_[object] = flat.empty() ? name : flat;
}

std::optional<int> Runtime::queueOfCounter(std::uint64_t signal) {
  const std::lock_guard lock(mutex_);
  for (const auto& [number, queue] : queues_) {
    if (queue->counterHandle() == signal) {
      return number;
    }
  }
  return std::nullopt;
}

hsa_signal_t Runtime::traceSignal() {
  {
    const std::lock_guard lock(traceMutex_);
    if (!traceSignals_.empty()) {
      const auto signal = traceSignals_.back();
      traceSignals_.pop_back();
      return signal;
    }
  }
  hsa_signal_t signal{};
  check(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal),
        "hsa_amd_signal_create");
  return signal;
}

void Runtime::freeTraceSignal(hsa_signal_t signal) {
  hsa_signal_store_relaxed(signal, 1);
  const std::lock_guard lock(traceMutex_);
  traceSignals_.push_back(signal);
}

namespace {
/**
 * Takes the times of the traced packets once they have completed, and hands them to the sink on
 * the clock of the host: the system timestamps of HSA, mapped onto std::chrono::steady_clock by
 * reading both together again and again.
 */
/**
 * DEVICE_OWN_QUEUE_DAG_LOG=<prefix>: while the kernels get traced (see trace()), what the streams
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
 *   (DEVICE_OWN_QUEUE_SCHEDULE), its predicted start and end, and its end on its queue alone
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
  std::string prefix_{utils::Env("DEVICE_").get<std::string>("OWN_QUEUE_DAG_LOG", "")};
  std::mutex mutex_;
  std::FILE* file_{nullptr};
  std::unordered_set<const void*> described_;
};

DagLog& dagLog() {
  static DagLog log;
  return log;
}

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

Tracer& tracer() {
  static Tracer tracer;
  return tracer;
}
} // namespace

void Runtime::finalize() {
  tracer().stop();
  hostThread.stop();
  const std::lock_guard lock(mutex_);
  queues_.clear();
  const std::lock_guard placeLock(placeMutex_);
  for (auto& pipe : pipes_) {
    hsa_queue_destroy(pipe.probe);
  }
  pipes_.clear();
  for (auto* queue : spareQueues_) {
    hsa_queue_destroy(queue);
  }
  spareQueues_.clear();
}

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

struct Event {
  hipEvent_t hip{nullptr};
  // recorded on an own stream: the mark of the queue
  Queue* queue{nullptr};
  std::uint64_t mark{0};
  // with timing, on an own stream: the signal of a kernel that took the time there
  bool timing{false};
  hsa_signal_t stamp{};
  bool stamped{false};
  // recorded on a stream of HIP (see Runtime::hipWaitsOnDevice): the counter of that stream (from
  // allocCounter()) and the value it gets after the work before the event
  std::uint32_t* hipCounter{nullptr};
  std::uint32_t hipValue{0};
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

/// the device can access the host memory: its address there, otherwise nullptr
const void* deviceAddress(const void* ptr) {
  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  if (hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS ||
      info.type == HSA_EXT_POINTER_TYPE_UNKNOWN) {
    return nullptr;
  }
  if (info.agentBaseAddress == nullptr || info.hostBaseAddress == nullptr ||
      info.agentBaseAddress == info.hostBaseAddress) {
    return ptr;
  }
  return static_cast<const char*>(info.agentBaseAddress) +
         (static_cast<const char*>(ptr) - static_cast<const char*>(info.hostBaseAddress));
}

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
  queue.dispatch(Queue::kernelOf(reinterpret_cast<const void*>(ownCopy)),
                 dim3(blocks),
                 dim3(CopyThreads),
                 0,
                 args,
                 fenceBits(acquire, release),
                 hsa_signal_t{0});
}

} // namespace

// ------------------------------------------------------------------------------------------------
// the translation of graphs

namespace {
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

/// the own queue number of helper lane `lane` of the programs of queue `queue`
int helperQueueNumber(int queue, unsigned lane) {
  return FirstHelperQueue + (queue + 1000) * 16 + static_cast<int>(lane);
}
} // namespace

/**
 * A graph translated for a queue. The kernels of a level of the graph are independent of each
 * other; a queue runs them one after the other though, also without barriers in between (MI250X,
 * ROCm 6.3). With DEVICE_OWN_QUEUE_GRAPH_LANES=k, the levels with at least
 * DEVICE_OWN_QUEUE_GRAPH_LANE_MIN kernels (3 by default) get spread over the queue and k helper
 * queues of its own (lanes), the largest kernels first onto the lane with the least so far; the
 * helpers start the level once the queue has reached it (a fork), and the queue goes on once the
 * helpers are done with it (a join), by signals of the program that each replay counts down. That
 * pays off only with pipes of the command processor left for the helpers (see
 * Runtime::placeQueue()): on the MI250X, with two own queues (TPV13 2.16 s instead of 2.74), not
 * with four (2.87 instead of 1.92). With DEVICE_OWN_QUEUE_GRAPH_BORROW=1, the lanes go to other own
 * queues instead, those that are idle at each replay (and only while the queue of the program is
 * idle as well, unless DEVICE_OWN_QUEUE_GRAPH_BORROW_IDLE=0); on four queues, that made no
 * difference (1.94 s), and coupled the queues without the condition (2.03 s).
 */
class Program {
  public:
  Program(Queue& queue, hipGraph_t graph);
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
  void waitOnLanes(hsa_signal_t signal, std::int64_t count, const std::vector<unsigned>& lanes);
  void push(unsigned lane, const PacketImage& image);

  Queue& queue_;
  std::vector<ProgramPacket> packets_;
  void* kernargs_{nullptr};
  std::uint64_t lastIndex_{0};
  // lane 0 is the queue of the program; the others are helper queues of their own, or (borrow_)
  // other own queues that are idle when the program gets enqueued, chosen anew each time
  std::vector<Queue*> lanes_;
  bool borrow_{false};
  // the queue of each lane in submit() (none: the lane runs on the queue of the program)
  std::vector<Queue*> laneQueues_;
  std::vector<hsa_signal_t> forks_;
  std::vector<std::vector<hsa_signal_t>> joins_;
  // how often each signal has been counted down so far
  std::vector<std::int64_t> forkCounts_;
  std::vector<std::vector<std::int64_t>> joinCounts_;

  // DEVICE_OWN_QUEUE_SCHEDULE (see submitScheduled()): the levels (packets first..first+count-1),
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
};

struct ProgramsImpl {
  std::mutex mutex;
  std::map<Queue*, std::unique_ptr<Program>> byQueue;
};

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

Program::Program(Queue& queue, hipGraph_t graph) : queue_(queue) {
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
  std::vector<std::size_t> pending(count, 0);
  for (std::size_t e = 0; e < edgeCount; ++e) {
    successors[index.at(from[e])].push_back(index.at(to[e]));
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
    logError() << "Own queues: a graph has a cycle.";
  }
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return level[a] < level[b];
  });

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
  };
  std::vector<Pending> work;
  std::size_t kernargBytes = 0;
  const auto addKernel = [&](const KernelInfo& kernel,
                             dim3 grid,
                             dim3 block,
                             std::size_t sharedMemBytes,
                             void** args,
                             std::size_t nodeLevel) {
    Pending item{&kernel, grid, block, sharedMemBytes, {}, 0, {}, nodeLevel};
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
        logError() << "Own queues: only plain copies are supported in graphs.";
      }
      auto* dst = reinterpret_cast<void*>(
          relocate(allocations, reinterpret_cast<std::uint64_t>(params.dstPtr.ptr)));
      const auto* src = reinterpret_cast<const void*>(
          relocate(allocations, reinterpret_cast<std::uint64_t>(params.srcPtr.ptr)));
      const auto* dstDevice = deviceAddress(dst);
      const auto* srcDevice = deviceAddress(src);
      if (dstDevice == nullptr || srcDevice == nullptr) {
        logError() << "Own queues: a graph copies from or to host memory that the device cannot "
                      "access (pageable memory).";
      }
      auto bytes = params.extent.width;
      const auto blocks = static_cast<std::uint32_t>(std::min<std::size_t>(
          MaxCopyBlocks, std::max<std::size_t>(1, bytes / (16 * CopyThreads))));
      void* args[] = {&dstDevice, &srcDevice, &bytes};
      addKernel(Queue::kernelOf(reinterpret_cast<const void*>(ownCopy)),
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
      addKernel(Queue::kernelOf(reinterpret_cast<const void*>(ownFill)),
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
      work.push_back(std::move(item));
      break;
    }
    case hipGraphNodeTypeEmpty:
    case hipGraphNodeTypeMemAlloc:
    case hipGraphNodeTypeMemFree:
      break;
    default:
      logError() << "Own queues: graph nodes of type" << static_cast<int>(type)
                 << "are not supported.";
    }
  }

  if (kernargBytes > 0) {
    kernargs_ = allocKernargMemory(kernargBytes);
  }
  // the helper lanes, and which level runs on which lanes (see Program)
  utils::Env env("DEVICE_");
  schedule_ = Runtime::instance().schedule;
  const auto laneCount = Runtime::instance().barrierValue && !schedule_
                             ? env.get<unsigned>("OWN_QUEUE_GRAPH_LANES", 0U)
                             : 0U;
  const auto laneMinimum = env.get<std::size_t>("OWN_QUEUE_GRAPH_LANE_MIN", std::size_t{3});
  borrow_ = laneCount > 0 && env.get<bool>("OWN_QUEUE_GRAPH_BORROW", false);
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
                 " may use printf, hostcalls, the device heap or device enqueue, which own "
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
  if (utils::Env("DEVICE_").get<bool>("OWN_QUEUE_VERBOSE", false)) {
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
    logInfo() << "Own queue" << queue.number() << ": a graph of" << count << "nodes and"
              << edgeCount << "edges became" << work.size() << "packets in"
              << kernelsPerLevel.size() << "levels of up to" << widest << "kernels;" << splitLevels
              << "of them spread over" << laneCount
              << (borrow_ ? "other own queues while idle" : "helper queues");
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
                          const std::vector<unsigned>& lanes) {
  // it has counted down `count` times once it is below the start minus the times before
  const auto below = CounterStart - count + 1;
  for (const auto lane : lanes) {
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
  laneQueues_.assign(lanes_.begin(), lanes_.end());
  std::vector<std::unique_lock<std::mutex>> borrowed;
  static std::atomic<std::uint64_t> submits{0};
  static std::atomic<std::uint64_t> lent{0};
  static const bool borrowWhenIdle =
      utils::Env("DEVICE_").get<bool>("OWN_QUEUE_GRAPH_BORROW_IDLE", true);
  static const bool verbose = utils::Env("DEVICE_").get<bool>("OWN_QUEUE_VERBOSE", false);
  const bool borrowNow = borrow_ && (!borrowWhenIdle || queue_.idleLocked());
  if (borrowNow) {
    // other own queues on other pipes that have nothing to do (with the queue of the program idle
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
      logInfo() << "Own queues: graphs borrowed" << lent.load() << "lanes in" << count << "replays";
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
      image.setHeader(packet.header, packet.rest);
      push(packet.lane, image);
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
 * Enqueues the program level by level (DEVICE_OWN_QUEUE_SCHEDULE): a level of independent kernels
 * goes onto the queue of the program, or, where that ends it earlier, partly onto own queues on
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
  static const bool verbose = utils::Env("DEVICE_").get<bool>("OWN_QUEUE_VERBOSE", false);
  if (verbose) {
    static std::atomic<std::uint64_t> replays{0};
    static std::atomic<std::uint64_t> splits{0};
    static std::atomic<std::uint64_t> moved{0};
    splits += splitLevels;
    moved += movedKernels;
    if (++replays % 20000 == 0) {
      logInfo() << "Own queues: in" << replays.load() << "replays, graphs spread" << splits.load()
                << "levels, with" << moved.load() << "kernels on other queues";
    }
  }
  lastIndex_ = queue_.created();
  return lastIndex_;
}

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
    logWarning() << "Own queues need the launch hook in the program (OwnQueueHook.cpp).";
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
    queue.waitForCounter(signal, value);
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
  queue.dispatch(Queue::kernelOf(reinterpret_cast<const void*>(ownWaitHost)),
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
// with the value of the latest event; the lock keeps the values in the order of the streams
struct HipStreamCounter {
  std::uint32_t* counter{nullptr};
  std::uint32_t value{0};
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
      hipLaunch(reinterpret_cast<const void*>(ownHipWaitBelow), dim3(1), dim3(1), args, 0, stream));
}
} // namespace

void* createEvent(bool withTiming) {
  auto* event = new Event();
  event->timing = withTiming && Runtime::instance().timing;
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
  if (isTagged(streamHandle) && !capturing(streamHandle)) {
    auto* stream = streamOf(streamHandle);
    auto& queue = *stream->queue;
    const std::lock_guard lock(queue.mutex);
    if (event->timing) {
      if (event->stamp.handle == 0) {
        check(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &event->stamp),
              "hsa_amd_signal_create");
      }
      queue.stamp(event->stamp);
      stream->lastIndex = queue.created();
      event->stamped = true;
    }
    event->queue = &queue;
    event->mark = queue.markCoveringLocked(stream->lastIndex);
    event->hipCounter = nullptr;
    if (dagLog().on()) {
      dagLog().line("R %lld %p %d %p %llu %llu\n",
                    DagLog::now(),
                    static_cast<void*>(stream),
                    queue.number(),
                    static_cast<void*>(event),
                    static_cast<unsigned long long>(event->mark),
                    static_cast<unsigned long long>(stream->lastIndex));
    }
    return;
  }
  // on a stream of HIP, or one that gets captured
  auto hipStream = vehicle(streamHandle);
  event->queue = nullptr;
  event->stamped = false;
  event->hipCounter = nullptr;
  internals::anchorCapture(hipStream);
  if (Runtime::instance().hipWaitsOnDevice && !isTagged(streamHandle) &&
      internals::captureState(hipStream).status != hipStreamCaptureStatusActive) {
    // own streams wait for the event by a counter that a kernel of ours sets after it (CLR does not
    // give the signal of an event)
    const std::lock_guard lock(hipCounterMutex);
    auto& counter = hipCounters[hipStream];
    if (counter.counter == nullptr) {
      counter.counter = allocCounter();
    }
    APIWRAP(hipEventRecord(event->hip, hipStream));
    internals::noteRecordedEvent(event->hip, hipStream);
    std::uint32_t* location = counter.counter;
    std::uint32_t value = ++counter.value;
    void* args[] = {&location, &value};
    APIWRAP(hipLaunch(
        reinterpret_cast<const void*>(ownWriteCounter), dim3(1), dim3(1), args, 0, hipStream));
    event->hipCounter = location;
    event->hipValue = value;
    return;
  }
  APIWRAP(hipEventRecord(event->hip, hipStream));
  internals::noteRecordedEvent(event->hip, hipStream);
}

double timespan(void* startHandle, void* endHandle) {
  auto* start = eventOf(startHandle);
  auto* end = eventOf(endHandle);
  if (start->stamped && end->stamped) {
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
  if (start->queue == nullptr && end->queue == nullptr && !start->stamped && !end->stamped) {
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
  event->queue = nullptr;
  event->stamped = false;
  event->hipCounter = nullptr;
  internals::forgetEvent(event->hip);
  APIWRAP(hipEventRecord(event->hip));
}

void streamWaitEvent(void* streamHandle, void* eventHandle) {
  if (isTagged(streamHandle) && !capturing(streamHandle)) {
    auto* stream = streamOf(streamHandle);
    if (!isTagged(eventHandle)) {
      warnOnce("own streams wait for events of HIP on the host");
      APIWRAP(hipEventSynchronize(static_cast<hipEvent_t>(eventHandle)));
      return;
    }
    auto* event = eventOf(eventHandle);
    if (event->queue == nullptr) {
      if (event->hipCounter != nullptr) {
        // the counter that a kernel of ours sets after the event on the stream of HIP
        waitMemory(streamHandle, event->hipCounter, event->hipValue);
        return;
      }
      warnOnce("own streams wait for events recorded on streams of HIP on the host");
      APIWRAP(hipEventSynchronize(event->hip));
      return;
    }
    auto& queue = *stream->queue;
    const std::lock_guard lock(queue.mutex);
    queue.waitFor(*event->queue, event->mark);
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
  if (isTagged(eventHandle) && eventOf(eventHandle)->queue != nullptr) {
    auto* event = eventOf(eventHandle);
    if (internals::captureState(hipStream).status == hipStreamCaptureStatusActive) {
      logError() << "Own queues: a stream that gets captured cannot wait for an own stream.";
    }
    if (Runtime::instance().hipWaitsOnDevice) {
      waitOnHipStream(*event->queue, event->mark, hipStream);
      return;
    }
    warnOnce("streams of HIP wait for events of own streams on the host");
    event->queue->wait(event->mark);
    return;
  }
  const auto hipEvent = hipEventOf(eventHandle);
  if (!internals::dependOnCapturedEvent(hipStream, hipEvent)) {
    APIWRAP(hipStreamWaitEvent(hipStream, hipEvent, 0));
  }
}

bool eventDone(void* handle) {
  auto* event = eventOf(handle);
  if (event->queue != nullptr) {
    return event->queue->reached(event->mark);
  }
  return APIWRAPX(hipEventQuery(event->hip), {hipErrorNotReady}) == hipSuccess;
}

void syncEvent(void* handle) {
  auto* event = eventOf(handle);
  if (event->queue != nullptr) {
    event->queue->wait(event->mark);
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
      entry = std::make_unique<Program>(queue, graph);
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

} // namespace device::ownqueues
