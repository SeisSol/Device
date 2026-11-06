// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

// Whether queues that wait (a barrier-value packet, a barrier-AND packet or a kernel that spins on
// a signal) slow down the kernels of another queue (gfx90a): a worker queue runs short kernels one
// after the other (one wave each, barrier bit, profiling), while P other queues wait for a signal
// that the host sets only afterwards; the durations of the worker's kernels and the gaps between
// them.
//
// hipcc -O2 --offload-arch=<arch> pollstall.cpp -lhsa-runtime64 -o pollstall
// pollstall [kernels] [microseconds per kernel]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <string>
#include <vector>

#define HIPCHECK(x)                                                                                \
  do {                                                                                             \
    hipError_t e_ = (x);                                                                           \
    if (e_ != hipSuccess) {                                                                        \
      std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__);           \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)
#define HSACHECK(x)                                                                                \
  do {                                                                                             \
    hsa_status_t s_ = (x);                                                                         \
    if (s_ != HSA_STATUS_SUCCESS) {                                                                \
      const char* t = nullptr;                                                                     \
      hsa_status_string(s_, &t);                                                                   \
      std::printf("HSA error %s at %s:%d\n", t ? t : "?", __FILE__, __LINE__);                     \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

// waits `cycles` of the constant clock (wall_clock64: 100 MHz on gfx90a, s_memrealtime there)
extern "C" __global__ void spin(std::uint64_t cycles) {
  const auto start = wall_clock64();
  while (wall_clock64() - start < cycles) {
    __builtin_amdgcn_s_sleep(1);
  }
}
extern "C" __global__ void tick(unsigned* counter) {}
extern "C" __global__ void waitBelowSlow(const volatile std::int64_t* value, std::int64_t below) {
  while (*value >= below) {
    __builtin_amdgcn_s_sleep(127);
  }
}
extern "C" __global__ void waitBelow(const volatile std::int64_t* value, std::int64_t below) {
  while (*value >= below) {
    __builtin_amdgcn_s_sleep(2);
  }
}

struct Kernel {
  std::uint64_t object{0};
  std::uint32_t kernargSize{0};
};

struct Hsa {
  hsa_agent_t gpu{}, cpu{};
  hsa_amd_memory_pool_t kernargPool{};
  bool havePool{false};
  std::uint32_t bus{0}, slot{0}, domain{0};
  Kernel tick, wait, waitSlow;
};

static hsa_status_t visitAgent(hsa_agent_t agent, void* data) {
  auto& hsa = *static_cast<Hsa*>(data);
  hsa_device_type_t type{};
  hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (type == HSA_DEVICE_TYPE_CPU && hsa.cpu.handle == 0) {
    hsa.cpu = agent;
  }
  if (type == HSA_DEVICE_TYPE_GPU) {
    std::uint32_t location = 0, domain = 0;
    hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &location);
    hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN), &domain);
    if (((location >> 8U) & 0xffU) == hsa.bus && ((location >> 3U) & 0x1fU) == hsa.slot &&
        domain == hsa.domain) {
      hsa.gpu = agent;
    }
  }
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t visitPool(hsa_amd_memory_pool_t pool, void* data) {
  auto& hsa = *static_cast<Hsa*>(data);
  std::uint32_t flags = 0;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0 && !hsa.havePool) {
    hsa.kernargPool = pool;
    hsa.havePool = true;
  }
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t
    visitSymbol(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t symbol, void* data) {
  auto& hsa = *static_cast<Hsa*>(data);
  hsa_symbol_kind_t kind{};
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind);
  if (kind != HSA_SYMBOL_KIND_KERNEL) {
    return HSA_STATUS_SUCCESS;
  }
  std::uint32_t length = 0;
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &length);
  std::string name(length, '\0');
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data());
  Kernel* kernel = nullptr;
  if (name == "spin.kd" || name == "spin") {
    kernel = &hsa.tick;
  }
  if (name == "waitBelow.kd" || name == "waitBelow") {
    kernel = &hsa.wait;
  }
  if (name == "waitBelowSlow.kd" || name == "waitBelowSlow") {
    kernel = &hsa.waitSlow;
  }
  if (kernel != nullptr) {
    hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel->object);
    hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernel->kernargSize);
  }
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t visitExecutable(hsa_executable_t executable, void* data) {
  hsa_executable_iterate_agent_symbols(executable, static_cast<Hsa*>(data)->gpu, visitSymbol, data);
  return HSA_STATUS_SUCCESS;
}

static std::uint64_t reserve(hsa_queue_t* q) {
  const auto index = hsa_queue_add_write_index_relaxed(q, 1);
  while (index - hsa_queue_load_read_index_scacquire(q) >= q->size) {
  }
  return index;
}

// writes the header last
static void publish(hsa_queue_t* q, void* slot, std::uint32_t headerAndSetup, std::uint64_t index) {
  __atomic_store_n(static_cast<std::uint32_t*>(slot), headerAndSetup, __ATOMIC_RELEASE);
  hsa_signal_store_screlease(q->doorbell_signal, static_cast<hsa_signal_value_t>(index));
}

static std::uint32_t fences(hsa_fence_scope_t acquire, hsa_fence_scope_t release) {
  return (static_cast<std::uint32_t>(acquire) << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
         (static_cast<std::uint32_t>(release) << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
}

constexpr std::uint32_t Barrier = 1U << HSA_PACKET_HEADER_BARRIER;

static void dispatch(hsa_queue_t* q,
                     const Kernel& kernel,
                     void* kernargs,
                     hsa_signal_t completion,
                     std::uint32_t fenceBits) {
  const auto index = reserve(q);
  auto* p = static_cast<hsa_kernel_dispatch_packet_t*>(q->base_address) + (index % q->size);
  std::memset(reinterpret_cast<std::uint8_t*>(p) + 4, 0, sizeof(*p) - 4);
  p->workgroup_size_x = 64;
  p->workgroup_size_y = 1;
  p->workgroup_size_z = 1;
  p->grid_size_x = 64;
  p->grid_size_y = 1;
  p->grid_size_z = 1;
  p->kernel_object = kernel.object;
  p->kernarg_address = kernargs;
  p->completion_signal = completion;
  const std::uint32_t header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) | Barrier | fenceBits;
  publish(q, p, header | (1U << 16U), index);
}

static void barrierAnd(hsa_queue_t* q,
                       hsa_signal_t dependency,
                       hsa_signal_t completion,
                       std::uint32_t fenceBits) {
  const auto index = reserve(q);
  auto* p = static_cast<hsa_barrier_and_packet_t*>(q->base_address) + (index % q->size);
  std::memset(reinterpret_cast<std::uint8_t*>(p) + 4, 0, sizeof(*p) - 4);
  p->dep_signal[0] = dependency;
  p->completion_signal = completion;
  const std::uint32_t header =
      (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | Barrier | fenceBits;
  publish(q, p, header, index);
}

// waits until the value of the signal is below `value` (AMD vendor packet)
static void barrierValueLt(hsa_queue_t* q,
                           hsa_signal_t signal,
                           std::int64_t value,
                           std::uint32_t fenceBits) {
  const auto index = reserve(q);
  auto* p = static_cast<hsa_amd_barrier_value_packet_t*>(q->base_address) + (index % q->size);
  std::memset(reinterpret_cast<std::uint8_t*>(p) + 4, 0, sizeof(*p) - 4);
  p->signal = signal;
  p->value = value;
  p->mask = -1;
  p->cond = HSA_SIGNAL_CONDITION_LT;
  const std::uint32_t header =
      (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) | Barrier | fenceBits;
  publish(
      q, p, header | (static_cast<std::uint32_t>(HSA_AMD_PACKET_TYPE_BARRIER_VALUE) << 16U), index);
}

static const char* phase = "";
static void waitBelowHost(hsa_signal_t s, std::int64_t value) {
  const auto start = std::chrono::steady_clock::now();
  while (hsa_signal_wait_scacquire(
             s, HSA_SIGNAL_CONDITION_LT, value, 1000000, HSA_WAIT_STATE_ACTIVE) >= value) {
    if (std::chrono::steady_clock::now() - start > std::chrono::seconds(20)) {
      std::printf("stuck in %s: a signal did not drop below %ld within 20 s\n",
                  phase,
                  static_cast<long>(value));
      std::exit(2);
    }
  }
}

static double us(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const int kernels = argc > 1 ? std::atoi(argv[1]) : 400;
  const double micro = argc > 2 ? std::atof(argv[2]) : 5.0;
  // kinds to run (a string of 0 barrier-value, 1 barrier-AND, 2 kernel), poller counts, priorities
  const std::string kindList = argc > 3 ? argv[3] : "012";
  std::vector<int> counts = {0, 1, 2, 4, 8, 16, 20};
  if (argc > 4) {
    counts.clear();
    for (const char* c = argv[4]; *c != 0;) {
      counts.push_back(std::atoi(c));
      while (*c != 0 && *c != ',')
        ++c;
      if (*c == ',')
        ++c;
    }
  }
  // worker priority and poller priority: l, n, h (or - to keep)
  const std::string priorities = argc > 5 ? argv[5] : "--";
  // how long the waits run before the worker starts (us), how many waiting queues get created
  const double settleMicro = argc > 6 ? std::atof(argv[6]) : 2000.0;
  const int created = argc > 7 ? std::atoi(argv[7]) : 20;
  // whether the worker's kernels carry the barrier bit (all of them, or only the first)
  // 1: all, 0: only the first, -1: none
  const int workerBarriers = argc > 8 ? std::atoi(argv[8]) : 1;
  // profiling: 1 worker only (default), 0 none (only the total time counts), 2 worker and waiting
  // queues
  const int profilingMode = argc > 9 ? std::atoi(argv[9]) : 1;
  // 1: the worker queue gets created after the waiting ones
  const bool workerLast = argc > 10 && std::atoi(argv[10]) != 0;
  Hsa hsa;
  int device = 0;
  HIPCHECK(hipGetDevice(&device));
  int bus = 0, slot = 0, domain = 0;
  HIPCHECK(hipDeviceGetAttribute(&bus, hipDeviceAttributePciBusId, device));
  HIPCHECK(hipDeviceGetAttribute(&slot, hipDeviceAttributePciDeviceId, device));
  HIPCHECK(hipDeviceGetAttribute(&domain, hipDeviceAttributePciDomainID, device));
  hsa.bus = bus;
  hsa.slot = slot;
  hsa.domain = domain;
  hipFuncAttributes attributes{};
  HIPCHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(spin)));
  HIPCHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(waitBelow)));
  HIPCHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(waitBelowSlow)));
  HSACHECK(hsa_init());
  HSACHECK(hsa_iterate_agents(visitAgent, &hsa));
  HSACHECK(hsa_amd_agent_iterate_memory_pools(hsa.cpu, visitPool, &hsa));
  hsa_ven_amd_loader_1_03_pfn_t loader{};
  HSACHECK(
      hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(loader), &loader));
  HSACHECK(loader.hsa_ven_amd_loader_iterate_executables(visitExecutable, &hsa));
  if (hsa.tick.object == 0 || hsa.wait.object == 0) {
    std::printf("kernels not found\n");
    return 1;
  }
  std::uint64_t frequency = 0;
  HSACHECK(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency));
  constexpr int MaxPollers = 20;
  void* args = nullptr;
  HSACHECK(hsa_amd_memory_pool_allocate(hsa.kernargPool, 256 * (MaxPollers + 2), 0, &args));
  HSACHECK(hsa_amd_agents_allow_access(1, &hsa.gpu, nullptr, args));
  std::memset(args, 0, 256 * (MaxPollers + 2));
  const auto cycles = static_cast<std::uint64_t>(micro * 100.0); // 100 MHz
  std::memcpy(args, &cycles, sizeof(cycles));
  hsa_queue_t* worker = nullptr;
  const auto createWorker = [&]() {
    HSACHECK(hsa_queue_create(hsa.gpu,
                              1 << 12,
                              HSA_QUEUE_TYPE_SINGLE,
                              nullptr,
                              nullptr,
                              UINT32_MAX,
                              UINT32_MAX,
                              &worker));
    HSACHECK(hsa_amd_profiling_set_profiler_enabled(worker, profilingMode > 0 ? 1 : 0));
  };
  if (!workerLast) {
    createWorker();
  }
  std::vector<hsa_queue_t*> pollers(std::min(created, MaxPollers), nullptr);
  for (auto*& q : pollers) {
    HSACHECK(hsa_queue_create(
        hsa.gpu, 1 << 8, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q));
    HSACHECK(hsa_amd_profiling_set_profiler_enabled(q, profilingMode == 2 ? 1 : 0));
  }
  if (workerLast) {
    createWorker();
  }
  const auto priorityOf = [](char c) {
    return c == 'l'   ? HSA_AMD_QUEUE_PRIORITY_LOW
           : c == 'h' ? HSA_AMD_QUEUE_PRIORITY_HIGH
                      : HSA_AMD_QUEUE_PRIORITY_NORMAL;
  };
  if (priorities[0] != '-') {
    HSACHECK(hsa_amd_queue_set_priority(worker, priorityOf(priorities[0])));
  }
  if (priorities[1] != '-') {
    for (auto* q : pollers) {
      HSACHECK(hsa_amd_queue_set_priority(q, priorityOf(priorities[1])));
    }
  }
  std::vector<hsa_signal_t> signals(kernels);
  for (auto& s : signals) {
    HSACHECK(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &s));
  }
  hsa_signal_t release{};
  HSACHECK(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &release));
  volatile hsa_signal_value_t* releaseValue = nullptr;
  HSACHECK(hsa_amd_signal_value_pointer(release, &releaseValue));
  std::vector<hsa_signal_t> drained(MaxPollers);
  for (auto& s : drained) {
    HSACHECK(hsa_signal_create(1, 0, nullptr, &s));
  }
  const hsa_signal_t none{0};
  const auto agent = fences(HSA_FENCE_SCOPE_AGENT, HSA_FENCE_SCOPE_AGENT);
  const auto dispatchOn = [&](hsa_queue_t* q,
                              const Kernel& kernel,
                              void* kernargs,
                              hsa_signal_t completion,
                              bool barrier = true) {
    const auto index = reserve(q);
    auto* p = static_cast<hsa_kernel_dispatch_packet_t*>(q->base_address) + (index % q->size);
    std::memset(reinterpret_cast<std::uint8_t*>(p) + 4, 0, sizeof(*p) - 4);
    p->workgroup_size_x = 64;
    p->workgroup_size_y = 1;
    p->workgroup_size_z = 1;
    p->grid_size_x = 64;
    p->grid_size_y = 1;
    p->grid_size_z = 1;
    p->kernel_object = kernel.object;
    p->kernarg_address = kernargs;
    p->completion_signal = completion;
    const std::uint32_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                                 (barrier ? Barrier : 0U) | agent;
    publish(q, p, header | (1U << 16U), index);
  };
  std::printf(
      "worker: %d kernels of %.1f us (one wave, barrier bit, profiling); P queues wait meanwhile\n",
      kernels,
      micro);
  std::printf("%-14s %3s %9s %8s %8s %8s %8s %8s %6s\n",
              "wait by",
              "P",
              "total us",
              "dur p50",
              "dur p90",
              "dur max",
              "gap p50",
              "gap p90",
              "long");
  const char* kinds[] = {"barrier-value",
                         "barrier-AND",
                         "kernel",
                         "long spin",
                         "slow kernel",
                         "spin no bit",
                         "spin no bit+bar",
                         "spin later",
                         "value later"};
  // the long spin of kind 3: 20000 times the worker's kernel
  void* longArgs = static_cast<char*>(args) + 256 * (MaxPollers + 1);
  const auto longCycles = cycles * 20000;
  std::memcpy(longArgs, &longCycles, sizeof(longCycles));
  for (const char kindChar : kindList) {
    const int kind = kindChar - '0';
    for (const int pollerCount : counts) {
      if (pollerCount > static_cast<int>(pollers.size())) {
        continue;
      }
      if (kind > 0 && pollerCount == 0) {
        continue;
      }
      for (int round = 0; round < 2; ++round) {
        hsa_signal_store_relaxed(release, 1);
        const auto startWaits = [&]() {
          for (int p = 0; p < pollerCount; ++p) {
            hsa_signal_store_relaxed(drained[p], 1);
            if (kind == 0 || kind == 8) {
              const auto index = reserve(pollers[p]);
              auto* packet =
                  static_cast<hsa_amd_barrier_value_packet_t*>(pollers[p]->base_address) +
                  (index % pollers[p]->size);
              std::memset(reinterpret_cast<std::uint8_t*>(packet) + 4, 0, sizeof(*packet) - 4);
              packet->signal = release;
              packet->value = 1;
              packet->mask = -1;
              packet->cond = HSA_SIGNAL_CONDITION_LT;
              packet->completion_signal = drained[p];
              const std::uint32_t header =
                  (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) | Barrier |
                  fences(HSA_FENCE_SCOPE_AGENT, HSA_FENCE_SCOPE_NONE);
              publish(pollers[p],
                      packet,
                      header |
                          (static_cast<std::uint32_t>(HSA_AMD_PACKET_TYPE_BARRIER_VALUE) << 16U),
                      index);
            } else if (kind == 1) {
              barrierAnd(pollers[p],
                         release,
                         drained[p],
                         fences(HSA_FENCE_SCOPE_AGENT, HSA_FENCE_SCOPE_NONE));
            } else if (kind == 3 || kind == 7) {
              dispatchOn(pollers[p], hsa.tick, longArgs, drained[p]);
            } else if (kind == 5) {
              // without the barrier bit
              dispatchOn(pollers[p], hsa.tick, longArgs, drained[p], false);
            } else if (kind == 6) {
              // without the barrier bit, without a completion signal, then a barrier-AND with it
              dispatchOn(pollers[p], hsa.tick, longArgs, none, false);
              barrierAnd(pollers[p],
                         none,
                         drained[p],
                         fences(HSA_FENCE_SCOPE_NONE, HSA_FENCE_SCOPE_AGENT));
            } else {
              auto* kernargs = static_cast<char*>(args) + 256 * (p + 1);
              const auto* value = const_cast<const std::int64_t*>(releaseValue);
              const std::int64_t below = 1;
              std::memcpy(kernargs, &value, sizeof(value));
              std::memcpy(kernargs + 8, &below, sizeof(below));
              dispatchOn(pollers[p], kind == 4 ? hsa.waitSlow : hsa.wait, kernargs, drained[p]);
            }
          }
        };
        const bool later = kind == 7 || kind == 8;
        if (!later) {
          startWaits();
        }
        // (the waits have started)
        const auto settle = std::chrono::steady_clock::now();
        while (us(settle, std::chrono::steady_clock::now()) < settleMicro) {
        }
        for (auto& s : signals)
          hsa_signal_store_relaxed(s, 1);
        phase = "the worker";
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < kernels; ++k) {
          dispatchOn(worker,
                     hsa.tick,
                     args,
                     signals[k],
                     workerBarriers == 1 || (workerBarriers == 0 && k == 0));
        }
        if (later) {
          const auto begin = std::chrono::steady_clock::now();
          while (us(begin, std::chrono::steady_clock::now()) < 1000.0) {
          }
          startWaits();
        }
        waitBelowHost(signals[kernels - 1], 1);
        const auto t1 = std::chrono::steady_clock::now();
        hsa_signal_store_screlease(release, 0);
        phase = "the release of the waits";
        for (int p = 0; p < pollerCount; ++p) {
          waitBelowHost(drained[p], 1);
        }
        std::vector<double> durations, gaps;
        double previousEnd = -1.0;
        int longest = 0;
        double longestDuration = 0.0;
        for (int k = 0; k < kernels && profilingMode > 0; ++k) {
          hsa_amd_profiling_dispatch_time_t time{};
          HSACHECK(hsa_amd_profiling_get_dispatch_time(hsa.gpu, signals[k], &time));
          const double start =
              1e6 * static_cast<double>(time.start) / static_cast<double>(frequency);
          const double end = 1e6 * static_cast<double>(time.end) / static_cast<double>(frequency);
          durations.push_back(end - start);
          if (end - start > longestDuration) {
            longestDuration = end - start;
            longest = k;
          }
          if (previousEnd >= 0.0)
            gaps.push_back(start - previousEnd);
          previousEnd = end;
        }
        if (profilingMode == 0) {
          std::printf("%-14s %3d %9.1f (no profiling)\n", kinds[kind], pollerCount, us(t0, t1));
          continue;
        }
        const auto longCount = std::count_if(
            durations.begin(), durations.end(), [&](double d) { return d > micro + 20.0; });
        std::sort(durations.begin(), durations.end());
        std::sort(gaps.begin(), gaps.end());
        const auto at = [](const std::vector<double>& v, double q) {
          return v[static_cast<std::size_t>(q * (v.size() - 1))];
        };
        std::printf("%-14s %3d %9.1f %8.2f %8.2f %8.1f %8.2f %8.2f %6ld  (longest: #%d)\n",
                    kinds[kind],
                    pollerCount,
                    us(t0, t1),
                    at(durations, 0.5),
                    at(durations, 0.9),
                    durations.back(),
                    at(gaps, 0.5),
                    at(gaps, 0.9),
                    static_cast<long>(longCount),
                    longest);
      }
    }
  }
  return 0;
}
