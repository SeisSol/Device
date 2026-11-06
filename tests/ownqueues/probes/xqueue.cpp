// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

// Latencies of dependencies on own HSA queues: a chain of tiny kernels on one queue (barrier bit),
// a ping-pong between two queues (barrier-AND on the completion signal of the other queue), a
// queue that waits for a value the host writes (barrier-value packet), and for comparison HIP
// streams with events.
//
// hipcc -O2 --offload-arch=<arch> xqueue.cpp -lhsa-runtime64 -o xqueue
#include <atomic>
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
#include <thread>
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

struct Hsa {
  hsa_agent_t gpu{}, cpu{};
  hsa_amd_memory_pool_t kernargPool{};
  bool havePool{false};
  std::uint32_t bus{0}, slot{0}, domain{0};
  std::uint64_t object{0};
  std::uint32_t kernargSize{0};
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
  if (name == "tick.kd" || name == "tick") {
    hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &hsa.object);
    hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &hsa.kernargSize);
  }
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t visitExecutable(hsa_executable_t executable, void* data) {
  hsa_executable_iterate_agent_symbols(executable, static_cast<Hsa*>(data)->gpu, visitSymbol, data);
  return HSA_STATUS_SUCCESS;
}

// writes a packet: all but the header, then the header
static std::uint64_t reserve(hsa_queue_t* q) {
  const auto index = hsa_queue_add_write_index_relaxed(q, 1);
  while (index - hsa_queue_load_read_index_scacquire(q) >= q->size) {
  }
  return index;
}

static void publish(hsa_queue_t* q, void* slot, std::uint32_t headerAndSetup, std::uint64_t index) {
  __atomic_store_n(static_cast<std::uint32_t*>(slot), headerAndSetup, __ATOMIC_RELEASE);
  hsa_signal_store_screlease(q->doorbell_signal, static_cast<hsa_signal_value_t>(index));
}

static constexpr std::uint32_t AgentFences =
    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

static void
    dispatch(Hsa& hsa, hsa_queue_t* q, void* kernargs, hsa_signal_t completion, bool barrier) {
  const auto index = reserve(q);
  auto* p = static_cast<hsa_kernel_dispatch_packet_t*>(q->base_address) + (index % q->size);
  std::memset(reinterpret_cast<std::uint8_t*>(p) + 4, 0, sizeof(*p) - 4);
  p->workgroup_size_x = 64;
  p->workgroup_size_y = 1;
  p->workgroup_size_z = 1;
  p->grid_size_x = 64;
  p->grid_size_y = 1;
  p->grid_size_z = 1;
  p->kernel_object = hsa.object;
  p->kernarg_address = kernargs;
  p->completion_signal = completion;
  const std::uint32_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                               ((barrier ? 1U : 0U) << HSA_PACKET_HEADER_BARRIER) | AgentFences;
  publish(q, p, header | (1U << 16U), index);
}

static void barrierAnd(hsa_queue_t* q, hsa_signal_t dependency, hsa_signal_t completion) {
  const auto index = reserve(q);
  auto* p = static_cast<hsa_barrier_and_packet_t*>(q->base_address) + (index % q->size);
  std::memset(reinterpret_cast<std::uint8_t*>(p) + 4, 0, sizeof(*p) - 4);
  p->dep_signal[0] = dependency;
  p->completion_signal = completion;
  const std::uint32_t header = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                               (1U << HSA_PACKET_HEADER_BARRIER) | AgentFences;
  publish(q, p, header, index);
}

// waits until the signal's value is below `value` (AMD vendor packet)
static void barrierValueLt(hsa_queue_t* q, hsa_signal_t signal, std::int64_t value) {
  const auto index = reserve(q);
  auto* p = static_cast<hsa_amd_barrier_value_packet_t*>(q->base_address) + (index % q->size);
  std::memset(reinterpret_cast<std::uint8_t*>(p) + 4, 0, sizeof(*p) - 4);
  p->signal = signal;
  p->value = value;
  p->mask = -1;
  p->cond = HSA_SIGNAL_CONDITION_LT;
  const std::uint32_t header = (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) |
                               (1U << HSA_PACKET_HEADER_BARRIER) |
                               (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
  const std::uint32_t amdFormat = HSA_AMD_PACKET_TYPE_BARRIER_VALUE;
  publish(q, p, header | (amdFormat << 16U), index);
}

// a mark: waits for all before it and counts the counter down
static void mark(hsa_queue_t* q, hsa_signal_t counter, bool systemRelease) {
  const auto index = reserve(q);
  auto* p = static_cast<hsa_barrier_and_packet_t*>(q->base_address) + (index % q->size);
  std::memset(reinterpret_cast<std::uint8_t*>(p) + 4, 0, sizeof(*p) - 4);
  p->completion_signal = counter;
  const std::uint32_t header = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                               (1U << HSA_PACKET_HEADER_BARRIER) |
                               ((systemRelease ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_AGENT)
                                << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  publish(q, p, header, index);
}

// waits until the signal's value is at least `value` (AMD vendor packet)
static void barrierValueGte(hsa_queue_t* q, hsa_signal_t signal, std::int64_t value) {
  const auto index = reserve(q);
  auto* p = static_cast<hsa_amd_barrier_value_packet_t*>(q->base_address) + (index % q->size);
  std::memset(reinterpret_cast<std::uint8_t*>(p) + 4, 0, sizeof(*p) - 4);
  p->signal = signal;
  p->value = value;
  p->mask = -1;
  p->cond = HSA_SIGNAL_CONDITION_GTE;
  const std::uint32_t header = (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) |
                               (1U << HSA_PACKET_HEADER_BARRIER) |
                               (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                               (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  const std::uint32_t amdFormat = HSA_AMD_PACKET_TYPE_BARRIER_VALUE;
  publish(q, p, header | (amdFormat << 16U), index);
}

static void waitBelow(hsa_signal_t s, std::int64_t value) {
  while (hsa_signal_wait_scacquire(
             s, HSA_SIGNAL_CONDITION_LT, value, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) >= value) {
  }
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 2000;
  // which parts: 1 chain, 2 ping-pong, 4 host wake, 8 HIP, 16 ping-pong with marks
  const int parts = argc > 2 ? std::atoi(argv[2]) : 31;
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
  hipFuncAttributes attributes{};
  HIPCHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(tick)));
  HSACHECK(hsa_init());
  HSACHECK(hsa_iterate_agents(visitAgent, &hsa));
  HSACHECK(hsa_amd_agent_iterate_memory_pools(hsa.cpu, visitPool, &hsa));
  hsa_ven_amd_loader_1_03_pfn_t loader{};
  HSACHECK(
      hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(loader), &loader));
  HSACHECK(loader.hsa_ven_amd_loader_iterate_executables(visitExecutable, &hsa));
  if (hsa.object == 0) {
    std::printf("kernel not found\n");
    return 1;
  }
  void* kernargs = nullptr;
  HSACHECK(hsa_amd_memory_pool_allocate(
      hsa.kernargPool, std::max<std::uint32_t>(hsa.kernargSize, 64), 0, &kernargs));
  HSACHECK(hsa_amd_agents_allow_access(1, &hsa.gpu, nullptr, kernargs));
  std::memset(kernargs, 0, std::max<std::uint32_t>(hsa.kernargSize, 64));
  std::memcpy(kernargs, &counter, sizeof(counter));

  hsa_queue_t *q1 = nullptr, *q2 = nullptr;
  HSACHECK(hsa_queue_create(
      hsa.gpu, 1 << 14, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q1));
  HSACHECK(hsa_queue_create(
      hsa.gpu, 1 << 14, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q2));
  hsa_signal_t done;
  HSACHECK(hsa_signal_create(1, 0, nullptr, &done));
  auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };

  for (int round = 0; round < 2; ++round) {
    // (1) chain on one queue: every kernel waits for the one before (barrier bit)
    hsa_signal_store_relaxed(done, 1);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n && (parts & 1) != 0; ++i) {
      dispatch(hsa, q1, kernargs, i + 1 == n ? done : hsa_signal_t{0}, true);
    }
    if ((parts & 1) != 0) {
      waitBelow(done, 1);
    }
    auto t1 = std::chrono::steady_clock::now();

    // (2) ping-pong between two queues: kernel on q1 -> barrier-AND on q2 -> kernel on q2 ->
    // barrier-AND on q1 -> ...; one signal per hop (created in advance)
    std::vector<hsa_signal_t> hop(n);
    for (auto& s : hop) {
      HSACHECK(hsa_signal_create(1, 0, nullptr, &s));
    }
    hsa_signal_store_relaxed(done, 1);
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < n && (parts & 2) != 0; ++i) {
      auto* q = i % 2 == 0 ? q1 : q2;
      if (i > 0) {
        barrierAnd(q, hop[i - 1], hsa_signal_t{0});
      }
      dispatch(hsa, q, kernargs, i + 1 == n ? done : hop[i], true);
    }
    if ((parts & 2) != 0) {
      waitBelow(done, 1);
    }
    auto t3 = std::chrono::steady_clock::now();
    for (auto& s : hop) {
      hsa_signal_destroy(s);
    }

    // (5) ping-pong as the own queues of the Device library do it: a kernel and a mark that counts
    // the counter of its queue down, and a barrier-value packet on the other queue that waits for
    // it
    double valueHop = 0;
    double valueHopAgent = 0;
    for (int variant = 0; variant < 2 && (parts & 16) != 0; ++variant) {
      const std::int64_t start = std::int64_t{1} << 62;
      hsa_signal_t counters[2];
      for (auto& c : counters) {
        HSACHECK(hsa_amd_signal_create(start, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &c));
      }
      std::int64_t marks[2] = {0, 0};
      hsa_signal_store_relaxed(done, 1);
      const auto a = std::chrono::steady_clock::now();
      for (int i = 0; i < n; ++i) {
        const int me = i % 2;
        auto* q = me == 0 ? q1 : q2;
        if (i > 0) {
          barrierValueLt(q, counters[1 - me], start - marks[1 - me] + 1);
        }
        dispatch(hsa, q, kernargs, i + 1 == n ? done : hsa_signal_t{0}, true);
        mark(q, counters[me], variant == 0);
        ++marks[me];
      }
      waitBelow(done, 1);
      const double hopTime = us(a, std::chrono::steady_clock::now()) / n;
      (variant == 0 ? valueHop : valueHopAgent) = hopTime;
      for (auto& c : counters) {
        hsa_signal_destroy(c);
      }
    }
    if ((parts & 16) != 0) {
      std::printf("round %d: ping-pong with marks and barrier-value packets: %.2f us/hop (system "
                  "release), %.2f us/hop (agent release)\n",
                  round,
                  valueHop,
                  valueHopAgent);
    }

    // (3) the host releases a waiting queue: barrier-value packets on a signal the host counts up,
    // each followed by a kernel that the host waits for, one at a time
    hsa_signal_t go;
    HSACHECK(hsa_signal_create(0, 0, nullptr, &go));
    hsa_signal_t step;
    HSACHECK(hsa_signal_create(0, 0, nullptr, &step));
    const int m = (parts & 4) != 0 ? std::min(n, 500) : 0;
    for (int i = 0; i < m; ++i) {
      barrierValueGte(q1, go, i + 1);
      dispatch(hsa, q1, kernargs, step, true);
    }
    double wake = 0;
    for (int i = 0; i < m; ++i) {
      hsa_signal_store_relaxed(step, 1);
      // let the queue settle in its wait
      const auto settle = std::chrono::steady_clock::now();
      while (us(settle, std::chrono::steady_clock::now()) < 20) {
      }
      const auto a = std::chrono::steady_clock::now();
      hsa_signal_store_screlease(go, i + 1);
      waitBelow(step, 1);
      wake += us(a, std::chrono::steady_clock::now());
    }
    hsa_signal_destroy(go);
    hsa_signal_destroy(step);

    // (4) HIP: chain on one stream, and ping-pong between two streams with events
    hipStream_t s1, s2;
    HIPCHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
    HIPCHECK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
    std::vector<hipEvent_t> events(n);
    for (auto& e : events) {
      HIPCHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
    }
    HIPCHECK(hipDeviceSynchronize());
    auto t4 = std::chrono::steady_clock::now();
    for (int i = 0; i < n && (parts & 8) != 0; ++i) {
      tick<<<1, 64, 0, s1>>>(counter);
    }
    HIPCHECK(hipStreamSynchronize(s1));
    auto t5 = std::chrono::steady_clock::now();
    for (int i = 0; i < n && (parts & 8) != 0; ++i) {
      auto s = i % 2 == 0 ? s1 : s2;
      if (i > 0) {
        HIPCHECK(hipStreamWaitEvent(s, events[i - 1], 0));
      }
      tick<<<1, 64, 0, s>>>(counter);
      HIPCHECK(hipEventRecord(events[i], s));
    }
    HIPCHECK(hipDeviceSynchronize());
    auto t6 = std::chrono::steady_clock::now();
    for (auto& e : events) {
      HIPCHECK(hipEventDestroy(e));
    }
    HIPCHECK(hipStreamDestroy(s1));
    HIPCHECK(hipStreamDestroy(s2));

    std::printf("round %d (%d kernels): own queue chain %.2f us/kernel; two own queues ping-pong "
                "%.2f us/hop; host wakes a waiting queue %.2f us (until its kernel is done); HIP "
                "stream chain %.2f us/kernel, two streams ping-pong %.2f us/hop\n",
                round,
                n,
                us(t0, t1) / n,
                us(t2, t3) / n,
                m > 0 ? wake / m : 0.0,
                us(t4, t5) / n,
                us(t5, t6) / n);
  }
  unsigned total = 0;
  HIPCHECK(hipMemcpy(&total, counter, sizeof(total), hipMemcpyDeviceToHost));
  std::printf("kernels run: %u\n", total);
  return 0;
}
