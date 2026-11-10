// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

// Latency of a dependency between two HSA queues, by the kind of wait: a ping-pong of tiny
// kernels, each waiting for the one before it on the other queue. All work gets enqueued behind a
// gate first, so that the time from opening the gate to the end is the device's alone.
//
//   and-kernel   the kernel completes a signal of its own (one per hop); the other queue waits for
//                it with a barrier-AND packet (what HIP does, without its markers)
//   and-mark     a mark after the kernel (barrier-AND without dependencies) completes a signal of
//                its own; the other queue waits for it with barrier-AND (a signal per mark)
//   value-mark   the mark counts a counter of its queue down; the other queue waits for it with a
//                barrier-value packet (the HSA queues of the Device library on gfx90a)
//   kernel-mark  as value-mark, but a kernel polls the counter (the HSA queues on gfx11)
//   hip          two streams with events, gated and not
//
// Signals with interrupts (hsa_signal_create) or without (GPU_ONLY); fences of the agent or the
// system for marks (release) and waits (acquire).
//
// hipcc -O2 --offload-arch=<arch> waitlat.cpp -lhsa-runtime64 -o waitlat
// waitlat [hops] [skip HIP: 1]
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

extern "C" __global__ void tick(unsigned* counter) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    atomicAdd(counter, 1U);
  }
}

// as hsaWaitBelow of the HSA queues
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
  Kernel tick, wait;
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
  if (name == "tick.kd" || name == "tick") {
    kernel = &hsa.tick;
  } else if (name == "waitBelow.kd" || name == "waitBelow") {
    kernel = &hsa.wait;
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

static void waitBelowHost(hsa_signal_t s, std::int64_t value) {
  while (hsa_signal_wait_scacquire(
             s, HSA_SIGNAL_CONDITION_LT, value, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) >= value) {
  }
}

enum class Wait { AndKernel, AndMark, ValueMark, KernelMark, Chain, ChainMark };

struct Variant {
  const char* name;
  Wait wait;
  bool interrupts;
  hsa_fence_scope_t markRelease;
  hsa_fence_scope_t waitAcquire;
};

static double us(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 2000;
  const bool skipHip = argc > 2 && std::atoi(argv[2]) == 1;
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
  unsigned* counter = nullptr;
  HIPCHECK(hipMalloc(&counter, sizeof(unsigned)));
  HIPCHECK(hipMemset(counter, 0, sizeof(unsigned)));
  // loads the code object
  hipFuncAttributes attributes{};
  HIPCHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(tick)));
  HIPCHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(waitBelow)));
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
  char agentName[64] = {};
  hsa_agent_get_info(hsa.gpu, HSA_AGENT_INFO_NAME, agentName);

  const auto stride = (std::max<std::uint32_t>(hsa.wait.kernargSize, 64) + 63U) / 64U * 64U;
  void* tickArgs = nullptr;
  HSACHECK(hsa_amd_memory_pool_allocate(
      hsa.kernargPool, std::max<std::uint32_t>(hsa.tick.kernargSize, 64), 0, &tickArgs));
  HSACHECK(hsa_amd_agents_allow_access(1, &hsa.gpu, nullptr, tickArgs));
  std::memset(tickArgs, 0, std::max<std::uint32_t>(hsa.tick.kernargSize, 64));
  std::memcpy(tickArgs, &counter, sizeof(counter));
  void* waitArgs = nullptr;
  HSACHECK(hsa_amd_memory_pool_allocate(hsa.kernargPool, stride * n, 0, &waitArgs));
  HSACHECK(hsa_amd_agents_allow_access(1, &hsa.gpu, nullptr, waitArgs));

  hsa_queue_t* queues[2] = {nullptr, nullptr};
  for (auto*& q : queues) {
    HSACHECK(hsa_queue_create(
        hsa.gpu, 1 << 15, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q));
  }
  hsa_signal_t done, gate;
  HSACHECK(hsa_signal_create(1, 0, nullptr, &done));
  HSACHECK(hsa_signal_create(1, 0, nullptr, &gate));
  const hsa_signal_t none{0};
  const auto kernelFences = fences(HSA_FENCE_SCOPE_AGENT, HSA_FENCE_SCOPE_AGENT);
  constexpr auto Agent = HSA_FENCE_SCOPE_AGENT;
  constexpr auto System = HSA_FENCE_SCOPE_SYSTEM;
  const Variant variants[] = {
      {"chain (one queue, kernels only)", Wait::Chain, false, Agent, Agent},
      {"chain with marks (one queue, system release)", Wait::ChainMark, false, System, Agent},
      {"and-kernel, GPU_ONLY signals, agent acquire", Wait::AndKernel, false, Agent, Agent},
      {"and-kernel, interrupt signals, agent acquire", Wait::AndKernel, true, Agent, Agent},
      {"and-kernel, interrupt signals, system acquire", Wait::AndKernel, true, Agent, System},
      {"and-mark, GPU_ONLY signals, agent/agent", Wait::AndMark, false, Agent, Agent},
      {"and-mark, GPU_ONLY signals, system/system", Wait::AndMark, false, System, System},
      {"and-mark, interrupt signals, system/system", Wait::AndMark, true, System, System},
      {"value-mark, agent/agent", Wait::ValueMark, false, Agent, Agent},
      {"value-mark, system/system (HSA queues, gfx90a)", Wait::ValueMark, false, System, System},
      {"kernel-mark, system/system (HSA queues, gfx11)", Wait::KernelMark, false, System, System},
  };
  constexpr std::int64_t Start = std::int64_t{1} << 62;
  std::printf("%s, %d hops; device time per hop after opening the gate (host enqueue time per "
              "hop), 3 rounds\n",
              agentName,
              n);

  for (const auto& variant : variants) {
    if (std::string(agentName).rfind("gfx9", 0) != 0 && variant.wait == Wait::ValueMark) {
      std::printf("%-50s skipped (no barrier-value packets)\n", variant.name);
      continue;
    }
    std::printf("%-50s", variant.name);
    for (int round = 0; round < 3; ++round) {
      std::vector<hsa_signal_t> hop(n, none);
      if (variant.wait == Wait::AndKernel || variant.wait == Wait::AndMark) {
        for (auto& s : hop) {
          if (variant.interrupts) {
            HSACHECK(hsa_signal_create(1, 0, nullptr, &s));
          } else {
            HSACHECK(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &s));
          }
        }
      }
      hsa_signal_t counters[2];
      volatile hsa_signal_value_t* counterValues[2] = {nullptr, nullptr};
      for (int c = 0; c < 2; ++c) {
        HSACHECK(
            hsa_amd_signal_create(Start, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &counters[c]));
        HSACHECK(hsa_amd_signal_value_pointer(counters[c], &counterValues[c]));
      }
      std::int64_t marks[2] = {0, 0};
      const auto markFences = fences(HSA_FENCE_SCOPE_NONE, variant.markRelease);
      const auto waitFences = fences(variant.waitAcquire, HSA_FENCE_SCOPE_NONE);
      hsa_signal_store_relaxed(done, 1);
      hsa_signal_store_relaxed(gate, 1);
      const bool oneQueue = variant.wait == Wait::Chain || variant.wait == Wait::ChainMark;
      const auto e0 = std::chrono::steady_clock::now();
      barrierAnd(queues[0], gate, none, fences(HSA_FENCE_SCOPE_SYSTEM, HSA_FENCE_SCOPE_NONE));
      int last = 0;
      for (int i = 0; i < n; ++i) {
        const int me = oneQueue ? 0 : i % 2;
        const int other = 1 - me;
        auto* q = queues[me];
        last = me;
        if (i > 0 && !oneQueue) {
          switch (variant.wait) {
          case Wait::AndKernel:
          case Wait::AndMark:
            barrierAnd(q, hop[i - 1], none, waitFences);
            break;
          case Wait::ValueMark:
            barrierValueLt(q, counters[other], Start - marks[other] + 1, waitFences);
            break;
          case Wait::KernelMark: {
            auto* args =
                static_cast<std::uint8_t*>(waitArgs) + static_cast<std::size_t>(i) * stride;
            std::memset(args, 0, stride);
            const std::int64_t below = Start - marks[other] + 1;
            const auto* value = const_cast<const hsa_signal_value_t*>(counterValues[other]);
            std::memcpy(args, &value, sizeof(value));
            std::memcpy(args + 8, &below, sizeof(below));
            dispatch(q, hsa.wait, args, none, waitFences);
            break;
          }
          default:
            break;
          }
        }
        dispatch(
            q, hsa.tick, tickArgs, variant.wait == Wait::AndKernel ? hop[i] : none, kernelFences);
        if (variant.wait == Wait::AndMark) {
          barrierAnd(q, none, hop[i], markFences);
        } else if (variant.wait == Wait::ValueMark || variant.wait == Wait::KernelMark ||
                   variant.wait == Wait::ChainMark) {
          barrierAnd(q, none, counters[me], markFences);
          ++marks[me];
        }
      }
      barrierAnd(queues[last], none, done, fences(HSA_FENCE_SCOPE_NONE, HSA_FENCE_SCOPE_SYSTEM));
      const auto e1 = std::chrono::steady_clock::now();
      const auto t0 = std::chrono::steady_clock::now();
      hsa_signal_store_screlease(gate, 0);
      waitBelowHost(done, 1);
      const auto t1 = std::chrono::steady_clock::now();
      std::printf("  %6.2f (%5.2f)", us(t0, t1) / n, us(e0, e1) / n);
      std::fflush(stdout);
      for (auto& s : hop) {
        if (s.handle != 0) {
          hsa_signal_destroy(s);
        }
      }
      for (auto& c : counters) {
        hsa_signal_destroy(c);
      }
    }
    std::printf(" us\n");
  }

  if (!skipHip) {
    // HIP: two streams with events; without gate (then, the host enqueues while the device runs),
    // and gated by a wait for a value in host memory (last: CLR may wait for resources behind it)
    hipStream_t s[2];
    for (auto& stream : s) {
      HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }
    std::vector<hipEvent_t> events(n);
    for (auto& e : events) {
      HIPCHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
    }
    std::uint32_t* gateHost = nullptr;
    HIPCHECK(hipHostMalloc(&gateHost, sizeof(std::uint32_t), hipHostMallocMapped));
    void* gateDevice = nullptr;
    HIPCHECK(hipHostGetDevicePointer(&gateDevice, gateHost, 0));
    for (int gated = 0; gated <= 1; ++gated) {
      std::printf("%-50s", gated == 1 ? "hip events, gated" : "hip events, no gate");
      for (int round = 0; round < 3; ++round) {
        HIPCHECK(hipDeviceSynchronize());
        __atomic_store_n(gateHost, 0U, __ATOMIC_RELEASE);
        const auto e0 = std::chrono::steady_clock::now();
        if (gated == 1) {
          HIPCHECK(hipStreamWaitValue32(s[0], gateDevice, 1, hipStreamWaitValueGte, 0xffffffff));
        }
        for (int i = 0; i < n; ++i) {
          auto stream = s[i % 2];
          if (i > 0) {
            HIPCHECK(hipStreamWaitEvent(stream, events[i - 1], 0));
          }
          tick<<<1, 64, 0, stream>>>(counter);
          HIPCHECK(hipEventRecord(events[i], stream));
        }
        const auto e1 = std::chrono::steady_clock::now();
        const auto t0 = std::chrono::steady_clock::now();
        __atomic_store_n(gateHost, 1U, __ATOMIC_RELEASE);
        HIPCHECK(hipStreamSynchronize(s[(n - 1) % 2]));
        const auto t1 = std::chrono::steady_clock::now();
        const double device = gated == 1 ? us(t0, t1) : us(e0, t1);
        std::printf("  %6.2f (%5.2f)", device / n, us(e0, e1) / n);
        std::fflush(stdout);
      }
      std::printf(" us\n");
    }
    for (auto& e : events) {
      HIPCHECK(hipEventDestroy(e));
    }
    for (auto& stream : s) {
      HIPCHECK(hipStreamDestroy(stream));
    }
    HIPCHECK(hipHostFree(gateHost));
  }
  unsigned total = 0;
  HIPCHECK(hipMemcpy(&total, counter, sizeof(total), hipMemcpyDeviceToHost));
  std::printf("kernels run: %u\n", total);
  return 0;
}
