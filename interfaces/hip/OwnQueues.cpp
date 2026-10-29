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
#include <cstring>
#include <cxxabi.h>
#include <deque>
#include <hip/hip_runtime.h>
#include <hsa/amd_hsa_queue.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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

__global__ void ownWaitHost(const volatile std::uint32_t* counter, std::uint32_t atLeast) {
  while (*counter < atLeast) {
    __builtin_amdgcn_s_sleep(2);
  }
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

// the marks of a queue count its counter down from here
constexpr std::int64_t CounterStart = std::int64_t{1} << 62;
// the most packets an own queue holds, and the default memory for the arguments of its kernels: the
// host must never wait for space while the queue waits for work the host has yet to enqueue (e.g.
// for halo data that MPI receives on the host), so both should hold the work of an epoch
constexpr std::uint32_t MaxQueuePackets = 1U << 17;
constexpr std::uint64_t DefaultKernargRingMiB = 128;
constexpr std::uint32_t CopyThreads = 256;
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
  HostThread hostThread;

  // the symbols of the kernels, from the loaded code objects; by their names (with .kd)
  std::unordered_map<std::string, KernelInfo> symbols;
  std::unordered_set<std::uint64_t> parsedCodeObjects;

  private:
  void scanCodeObjects();

  std::mutex mutex_;
  std::unordered_map<const void*, const KernelInfo*> kernels_;
  std::map<int, std::unique_ptr<Queue>> queues_;
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
  Queue(Runtime& runtime, int number) : runtime_(runtime), number_(number) {
    check(hsa_queue_create(runtime.gpu,
                           runtime.queuePackets,
                           HSA_QUEUE_TYPE_MULTI,
                           queueError,
                           nullptr,
                           UINT32_MAX,
                           UINT32_MAX,
                           &queue_),
          "hsa_queue_create");
    check(hsa_amd_signal_create(CounterStart, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &counter_),
          "hsa_amd_signal_create");
    check(hsa_amd_signal_value_pointer(counter_, &counterValue_), "hsa_amd_signal_value_pointer");
    if (runtime.timing || runtime.tracing.load()) {
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
  const std::lock_guard lock(mutex_);
  auto& queue = queues_[number];
  if (queue == nullptr) {
    queue = std::make_unique<Queue>(*this, number);
  }
  return *queue;
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
}

// ------------------------------------------------------------------------------------------------
// streams and events

struct Stream {
  Queue* queue{nullptr};
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
};
} // namespace

class Program {
  public:
  Program(Queue& queue, hipGraph_t graph);
  ~Program();

  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  /// enqueues the program on its queue, with the queue's mutex held; returns the write index after
  std::uint64_t submit();

  private:
  Queue& queue_;
  std::vector<ProgramPacket> packets_;
  void* kernargs_{nullptr};
  std::uint64_t lastIndex_{0};
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
  std::size_t previousLevel = SIZE_MAX;
  for (auto& item : work) {
    ProgramPacket packet;
    // the first one of each level waits for all before it (and so for the work before the graph)
    const bool barrier = item.level != previousLevel;
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
  }
  if (kernargBytes > 0) {
    flushKernargs(static_cast<const char*>(kernargs_) + kernargBytes - 1);
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
}

std::uint64_t Program::submit() {
  for (const auto& packet : packets_) {
    if (packet.host) {
      queue_.hostFunction(packet.host);
    } else {
      PacketImage image;
      image.as<hsa_kernel_dispatch_packet_t>() = packet.packet;
      image.setHeader(packet.header, packet.rest);
      queue_.push(image);
    }
  }
  lastIndex_ = queue_.created();
  return lastIndex_;
}

// ------------------------------------------------------------------------------------------------
// the interface

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
  queue.dispatch(kernel, grid, block, sharedMemBytes, args, KernelFences, hsa_signal_t{0});
  stream->lastIndex = queue.created();
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
  queue.hostFunction(function);
  stream->lastIndex = queue.created();
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
    queue.waitForCounter(signal, value);
    stream->lastIndex = queue.created();
    return;
  }
  std::uint32_t* deviceLocation = location;
  if (signal.handle == 0) {
    APIWRAP(hipHostGetDevicePointer(reinterpret_cast<void**>(&deviceLocation), location, 0));
  }
  const std::uint32_t* counter = deviceLocation;
  void* args[] = {&counter, &value};
  const std::lock_guard lock(queue.mutex);
  queue.dispatch(Queue::kernelOf(reinterpret_cast<const void*>(ownWaitHost)),
                 dim3(1),
                 dim3(1),
                 0,
                 args,
                 queue.hostWaitFences(),
                 hsa_signal_t{0});
  queue.acquireFromHost();
  stream->lastIndex = queue.created();
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
    dispatchCopy(queue, dstDevice, staging, bytes);
    stream->lastIndex = queue.created();
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
  dispatchCopy(queue, dstDevice, srcDevice, bytes);
  stream->lastIndex = queue.created();
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
    return;
  }
  // on a stream of HIP, or one that gets captured
  auto hipStream = vehicle(streamHandle);
  event->queue = nullptr;
  event->stamped = false;
  internals::anchorCapture(hipStream);
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

bool trace(const TraceSink& sink) {
  if (!enabled()) {
    return false;
  }
  if (sink) {
    tracer().start(sink);
  } else {
    tracer().stop();
  }
  return true;
}

void recordEventOnHost(void* handle) {
  auto* event = eventOf(handle);
  event->queue = nullptr;
  event->stamped = false;
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
      warnOnce("own streams wait for events recorded on streams of HIP on the host");
      APIWRAP(hipEventSynchronize(event->hip));
      return;
    }
    auto& queue = *stream->queue;
    const std::lock_guard lock(queue.mutex);
    queue.waitFor(*event->queue, event->mark);
    stream->lastIndex = queue.created();
    return;
  }
  auto hipStream = vehicle(streamHandle);
  if (isTagged(eventHandle) && eventOf(eventHandle)->queue != nullptr) {
    auto* event = eventOf(eventHandle);
    if (internals::captureState(hipStream).status == hipStreamCaptureStatusActive) {
      logError() << "Own queues: a stream that gets captured cannot wait for an own stream.";
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
  stream->lastIndex = program->submit();
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
