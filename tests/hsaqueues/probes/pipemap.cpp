// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

// Whether the pipes of the command processor that queues get stay put when other queues go
// (gfx90a): four reference queues, then test queues; each test queue's pipe is the reference it
// shares with (a long kernel on the test queue delays a short one that a reference starts after
// it); then some test queues get destroyed, the others checked again, and new ones placed.
//
// hipcc -O2 --offload-arch=<arch> pipemap.cpp -lhsa-runtime64 -o pipemap
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

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
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
  HSACHECK(hsa_init());
  HSACHECK(hsa_iterate_agents(visitAgent, &hsa));
  HSACHECK(hsa_amd_agent_iterate_memory_pools(hsa.cpu, visitPool, &hsa));
  hsa_ven_amd_loader_1_03_pfn_t loader{};
  HSACHECK(
      hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(loader), &loader));
  HSACHECK(loader.hsa_ven_amd_loader_iterate_executables(visitExecutable, &hsa));
  void* longArgs = nullptr;
  void* shortArgs = nullptr;
  HSACHECK(hsa_amd_memory_pool_allocate(hsa.kernargPool, 256, 0, &longArgs));
  HSACHECK(hsa_amd_memory_pool_allocate(hsa.kernargPool, 256, 0, &shortArgs));
  HSACHECK(hsa_amd_agents_allow_access(1, &hsa.gpu, nullptr, longArgs));
  HSACHECK(hsa_amd_agents_allow_access(1, &hsa.gpu, nullptr, shortArgs));
  std::memset(longArgs, 0, 256);
  std::memset(shortArgs, 0, 256);
  const std::uint64_t longCycles = 30000, shortCycles = 1;
  std::memcpy(longArgs, &longCycles, 8);
  std::memcpy(shortArgs, &shortCycles, 8);
  const auto create = [&]() {
    hsa_queue_t* q = nullptr;
    HSACHECK(hsa_queue_create(
        hsa.gpu, 1 << 12, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q));
    return q;
  };
  const auto dispatchOn = [&](hsa_queue_t* q, void* kernargs, hsa_signal_t completion) {
    const auto index = reserve(q);
    auto* p = static_cast<hsa_kernel_dispatch_packet_t*>(q->base_address) + (index % q->size);
    std::memset(reinterpret_cast<std::uint8_t*>(p) + 4, 0, sizeof(*p) - 4);
    p->workgroup_size_x = 64;
    p->workgroup_size_y = 1;
    p->workgroup_size_z = 1;
    p->grid_size_x = 64;
    p->grid_size_y = 1;
    p->grid_size_z = 1;
    p->kernel_object = hsa.tick.object;
    p->kernarg_address = kernargs;
    p->completion_signal = completion;
    const std::uint32_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                                 Barrier | fences(HSA_FENCE_SCOPE_AGENT, HSA_FENCE_SCOPE_AGENT);
    publish(q, p, header | (1U << 16U), index);
  };
  hsa_signal_t a{}, b{};
  HSACHECK(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &a));
  HSACHECK(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &b));
  double longMicro = 0;
  {
    // how long the long kernel takes alone
    auto* q = create();
    hsa_signal_store_relaxed(a, 1);
    const auto t0 = std::chrono::steady_clock::now();
    dispatchOn(q, longArgs, a);
    waitBelowHost(a, 1);
    longMicro = us(t0, std::chrono::steady_clock::now());
    HSACHECK(hsa_queue_destroy(q));
  }
  std::printf("the long kernel takes %.0f us alone\n", longMicro);
  const auto samePipe = [&](hsa_queue_t* first, hsa_queue_t* second) {
    hsa_signal_store_relaxed(a, 1);
    hsa_signal_store_relaxed(b, 1);
    dispatchOn(first, longArgs, a);
    const auto settle = std::chrono::steady_clock::now();
    while (us(settle, std::chrono::steady_clock::now()) < 30.0) {
    }
    const auto t0 = std::chrono::steady_clock::now();
    dispatchOn(second, shortArgs, b);
    waitBelowHost(b, 1);
    const double took = us(t0, std::chrono::steady_clock::now());
    waitBelowHost(a, 1);
    return took > longMicro / 2;
  };
  std::vector<hsa_queue_t*> references;
  while (true) {
    auto* q = create();
    bool known = false;
    for (auto* r : references) {
      if (samePipe(q, r)) {
        known = true;
        break;
      }
    }
    if (known) {
      HSACHECK(hsa_queue_destroy(q));
      break;
    }
    references.push_back(q);
  }
  std::printf("%zu pipes\n", references.size());
  const auto pipeOf = [&](hsa_queue_t* q) {
    for (std::size_t r = 0; r < references.size(); ++r) {
      if (samePipe(q, references[r]))
        return static_cast<int>(r);
    }
    return -1;
  };
  std::vector<hsa_queue_t*> tests;
  std::printf("test queues 0-9 on pipes:");
  for (int t = 0; t < 10; ++t) {
    tests.push_back(create());
    std::printf(" %d", pipeOf(tests.back()));
  }
  std::printf("\n");
  for (int t : {1, 2, 6}) {
    HSACHECK(hsa_queue_destroy(tests[t]));
    tests[t] = nullptr;
  }
  std::printf("after destroying 1, 2, 6:");
  for (int t = 0; t < 10; ++t) {
    if (tests[t] == nullptr)
      std::printf(" -");
    else
      std::printf(" %d", pipeOf(tests[t]));
  }
  std::printf("\nnew queues 10-13 on pipes:");
  for (int t = 0; t < 4; ++t) {
    tests.push_back(create());
    std::printf(" %d", pipeOf(tests.back()));
  }
  std::printf("\nall again:");
  for (std::size_t t = 0; t < tests.size(); ++t) {
    if (tests[t] == nullptr)
      std::printf(" -");
    else
      std::printf(" %d", pipeOf(tests[t]));
  }
  std::printf("\n");
  return 0;
}
