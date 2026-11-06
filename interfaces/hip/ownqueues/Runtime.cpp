// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Runtime.h"

#include "Common.h"
#include "Internals.h"
#include "Kernels.h"
#include "Queue.h"
#include "Trace.h"
#include "utils/env.h"
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
#include <cxxabi.h>
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

namespace {

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

void queueError(hsa_status_t status, hsa_queue_t* queue, void* /*data*/) {
  const char* text = nullptr;
  hsa_status_string(status, &text);
  logError() << "Own queues: the queue" << static_cast<const void*>(queue) << "failed with"
             << (text != nullptr ? text : "?");
}

} // namespace

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

namespace {

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

} // namespace

Runtime& Runtime::instance() {
  static Runtime runtime;
  return runtime;
}

Runtime::Runtime() = default;
Runtime::~Runtime() = default;

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
  // where the kernel arguments go: device memory that the host writes, or host memory. On MI250X
  // and MI300 (gfx90a, gfx94x), device memory by default: on an MI300A, TPV5 and TPV13 took 9-10 %
  // less on four ranks, TPV5 8 % less on one, and 21 % less with recorded super-timesteps (where
  // the kernels read the same arguments again and again); on two nodes of MI250X, TPV13 1.7 % less
  // (TPV5 the same).
  const bool deviceByDefault = arch == "gfx90a" || arch.rfind("gfx94", 0) == 0;
  const auto kernargs =
      env.get<std::string>("OWN_QUEUE_KERNARGS", deviceByDefault ? "device" : "host");
  if (kernargs == "device") {
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
  deviceName_ = arch;
  return true;
}

void Runtime::describe() {
  logInfo() << "Own queues on" << deviceName_ << "; queues wait for other queues by"
            << (barrierValue ? "barrier-value packets" : "kernels") << "; work after the host sees"
            << "its data by" << (acquireByKernel ? "a fence in a kernel" : "packet fences")
            << "; kernel arguments in" << (deviceKernargs ? "device memory" : "host memory")
            << "; queues of" << queuePackets << "packets and" << (kernargRingBytes >> 20)
            << "MiB for kernel arguments";
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
    if (!described_) {
      described_ = true;
      describe();
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

} // namespace device::ownqueues
