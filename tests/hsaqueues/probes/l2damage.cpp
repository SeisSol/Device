// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

// What fences of the system scope cost the kernels of other queues: a chain of kernels on one
// queue reads a buffer that fits into the L2 cache, while another queue runs a chain of fences of
// one kind, as fast as it can. All work gets enqueued behind a gate first.
//
//   none            the other queue idles
//   packet-acquire  barrier-AND packets with an acquire fence (agent, system), as the HSA queues
//                   wait for other queues and for the host
//   packet-release  barrier-AND packets with a release fence (agent, system), as marks
//   kernel-acquire  a tiny kernel (packet fences of the agent) that fences in the shader:
//                   __builtin_amdgcn_fence(acquire, agent / system), i.e. buffer_invl2 for the
//                   system on gfx90a, which only drops lines of non-local memory
//   kernel-release  as kernel-acquire with a release fence (buffer_wbl2 for the system)
//
// hipcc -O2 --offload-arch=<arch> l2damage.cpp -lhsa-runtime64 -o l2damage
// l2damage [MiB read per kernel pass] [passes per kernel] [kernels]
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

// reads `count` float4 `passes` times (the L2 keeps them if they fit), with `threads` threads in
// blocks of 256 (gridDim and blockDim come from hidden arguments, which the dispatches leave 0)
extern "C" __global__ void
    victim(const float4* data, std::uint64_t count, int passes, float* out, std::uint64_t threads) {
  float sum = 0;
  for (int pass = 0; pass < passes; ++pass) {
    for (std::uint64_t i = blockIdx.x * 256 + threadIdx.x; i < count; i += threads) {
      const float4 v = data[i];
      sum += v.x + v.y + v.z + v.w;
    }
  }
  if (sum == 12345.0f) {
    out[0] = sum;
  }
}

extern "C" __global__ void tinyAcquireSystem() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""); }
extern "C" __global__ void tinyAcquireAgent() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); }
extern "C" __global__ void tinyReleaseSystem() { __builtin_amdgcn_fence(__ATOMIC_RELEASE, ""); }
extern "C" __global__ void tinyReleaseAgent() { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent"); }
extern "C" __global__ void tinyNothing() {}

struct Kernel {
  std::uint64_t object{0};
  std::uint32_t kernargSize{0};
};

struct Hsa {
  hsa_agent_t gpu{}, cpu{};
  hsa_amd_memory_pool_t kernargPool{};
  bool havePool{false};
  std::uint32_t bus{0}, slot{0}, domain{0};
  Kernel victim, acquireSystem, acquireAgent, releaseSystem, releaseAgent, nothing;
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
  const auto is = [&](const char* kernelName) {
    return name == std::string(kernelName) || name == std::string(kernelName) + ".kd";
  };
  if (is("victim")) {
    kernel = &hsa.victim;
  } else if (is("tinyAcquireSystem")) {
    kernel = &hsa.acquireSystem;
  } else if (is("tinyAcquireAgent")) {
    kernel = &hsa.acquireAgent;
  } else if (is("tinyReleaseSystem")) {
    kernel = &hsa.releaseSystem;
  } else if (is("tinyReleaseAgent")) {
    kernel = &hsa.releaseAgent;
  } else if (is("tinyNothing")) {
    kernel = &hsa.nothing;
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
                     std::uint32_t fenceBits,
                     std::uint32_t blocks = 1) {
  const auto index = reserve(q);
  auto* p = static_cast<hsa_kernel_dispatch_packet_t*>(q->base_address) + (index % q->size);
  std::memset(reinterpret_cast<std::uint8_t*>(p) + 4, 0, sizeof(*p) - 4);
  p->workgroup_size_x = 256;
  p->workgroup_size_y = 1;
  p->workgroup_size_z = 1;
  p->grid_size_x = 256 * blocks;
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

enum class Kind { None, PacketAcquire, PacketRelease, KernelAcquire, KernelRelease, KernelNothing };

struct Variant {
  const char* name;
  Kind kind;
  hsa_fence_scope_t scope;
};

static double us(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

int main(int argc, char** argv) {
  const double mib = argc > 1 ? std::atof(argv[1]) : 4.0;
  const int passes = argc > 2 ? std::atoi(argv[2]) : 16;
  const int kernels = argc > 3 ? std::atoi(argv[3]) : 500;
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
  const auto count = static_cast<std::uint64_t>(mib * 1024 * 1024 / sizeof(float4));
  float4* data = nullptr;
  HIPCHECK(hipMalloc(&data, count * sizeof(float4)));
  HIPCHECK(hipMemset(data, 0, count * sizeof(float4)));
  float* out = nullptr;
  HIPCHECK(hipMalloc(&out, sizeof(float)));
  hipFuncAttributes attributes{};
  for (const void* f : {reinterpret_cast<const void*>(victim),
                        reinterpret_cast<const void*>(tinyAcquireSystem),
                        reinterpret_cast<const void*>(tinyAcquireAgent),
                        reinterpret_cast<const void*>(tinyReleaseSystem),
                        reinterpret_cast<const void*>(tinyReleaseAgent),
                        reinterpret_cast<const void*>(tinyNothing)}) {
    HIPCHECK(hipFuncGetAttributes(&attributes, f));
  }
  HIPCHECK(hipDeviceSynchronize());
  HSACHECK(hsa_init());
  HSACHECK(hsa_iterate_agents(visitAgent, &hsa));
  HSACHECK(hsa_amd_agent_iterate_memory_pools(hsa.cpu, visitPool, &hsa));
  hsa_ven_amd_loader_1_03_pfn_t loader{};
  HSACHECK(
      hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(loader), &loader));
  HSACHECK(loader.hsa_ven_amd_loader_iterate_executables(visitExecutable, &hsa));
  for (const Kernel* k : {&hsa.victim,
                          &hsa.acquireSystem,
                          &hsa.acquireAgent,
                          &hsa.releaseSystem,
                          &hsa.releaseAgent,
                          &hsa.nothing}) {
    if (k->object == 0) {
      std::printf("kernels not found\n");
      return 1;
    }
  }
  char agentName[64] = {};
  hsa_agent_get_info(hsa.gpu, HSA_AGENT_INFO_NAME, agentName);
  std::uint32_t computeUnits = 0;
  hsa_agent_get_info(
      hsa.gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT), &computeUnits);

  const std::uint32_t blocks = 4 * computeUnits;
  struct Args {
    const float4* data;
    std::uint64_t count;
    int passes;
    int pad;
    float* out;
    std::uint64_t threads;
  };
  void* victimArgs = nullptr;
  const auto victimArgBytes = std::max<std::uint32_t>(hsa.victim.kernargSize, 64);
  HSACHECK(hsa_amd_memory_pool_allocate(hsa.kernargPool, victimArgBytes, 0, &victimArgs));
  HSACHECK(hsa_amd_agents_allow_access(1, &hsa.gpu, nullptr, victimArgs));
  std::memset(victimArgs, 0, victimArgBytes);
  const Args args{data, count, passes, 0, out, 256ULL * blocks};
  std::memcpy(victimArgs, &args, sizeof(args));
  void* tinyArgs = nullptr;
  HSACHECK(hsa_amd_memory_pool_allocate(hsa.kernargPool, 512, 0, &tinyArgs));
  HSACHECK(hsa_amd_agents_allow_access(1, &hsa.gpu, nullptr, tinyArgs));
  std::memset(tinyArgs, 0, 512);

  hsa_queue_t* queues[2] = {nullptr, nullptr};
  for (auto*& q : queues) {
    HSACHECK(hsa_queue_create(
        hsa.gpu, 1 << 17, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q));
  }
  hsa_signal_t victimDone, otherDone, gate;
  HSACHECK(hsa_signal_create(1, 0, nullptr, &victimDone));
  HSACHECK(hsa_signal_create(1, 0, nullptr, &otherDone));
  HSACHECK(hsa_signal_create(1, 0, nullptr, &gate));
  const hsa_signal_t none{0};
  const auto kernelFences = fences(HSA_FENCE_SCOPE_AGENT, HSA_FENCE_SCOPE_AGENT);
  const Variant variants[] = {
      {"none", Kind::None, HSA_FENCE_SCOPE_AGENT},
      {"tiny kernels, no fence", Kind::KernelNothing, HSA_FENCE_SCOPE_AGENT},
      {"packet acquire, agent", Kind::PacketAcquire, HSA_FENCE_SCOPE_AGENT},
      {"packet acquire, system", Kind::PacketAcquire, HSA_FENCE_SCOPE_SYSTEM},
      {"packet release, agent", Kind::PacketRelease, HSA_FENCE_SCOPE_AGENT},
      {"packet release, system", Kind::PacketRelease, HSA_FENCE_SCOPE_SYSTEM},
      {"kernel acquire, agent", Kind::KernelAcquire, HSA_FENCE_SCOPE_AGENT},
      {"kernel acquire, system (buffer_invl2)", Kind::KernelAcquire, HSA_FENCE_SCOPE_SYSTEM},
      {"kernel release, agent", Kind::KernelRelease, HSA_FENCE_SCOPE_AGENT},
      {"kernel release, system (buffer_wbl2)", Kind::KernelRelease, HSA_FENCE_SCOPE_SYSTEM},
  };
  std::printf("%s, %u CUs; victim: %u blocks read %.1f MiB %d times per kernel, %d kernels; "
              "us per victim kernel (fences per victim kernel), 3 rounds\n",
              agentName,
              computeUnits,
              blocks,
              mib,
              passes,
              kernels);
  // the other queue: enough fences to outlast the victims, but fewer than its queue holds (all
  // get enqueued behind the gate)
  const int others = 120000;
  for (const auto& variant : variants) {
    std::printf("%-42s", variant.name);
    for (int round = 0; round < 3; ++round) {
      hsa_signal_store_relaxed(victimDone, 1);
      hsa_signal_store_relaxed(otherDone, 1);
      hsa_signal_store_relaxed(gate, 1);
      barrierAnd(queues[0], gate, none, fences(HSA_FENCE_SCOPE_SYSTEM, HSA_FENCE_SCOPE_NONE));
      for (int i = 0; i < kernels; ++i) {
        dispatch(queues[0],
                 hsa.victim,
                 victimArgs,
                 i + 1 == kernels ? victimDone : none,
                 kernelFences,
                 blocks);
      }
      int enqueued = 0;
      if (variant.kind != Kind::None) {
        barrierAnd(queues[1], gate, none, fences(HSA_FENCE_SCOPE_SYSTEM, HSA_FENCE_SCOPE_NONE));
        for (int i = 0; i < others; ++i) {
          switch (variant.kind) {
          case Kind::PacketAcquire:
            barrierAnd(queues[1], none, none, fences(variant.scope, HSA_FENCE_SCOPE_NONE));
            break;
          case Kind::PacketRelease:
            barrierAnd(queues[1], none, none, fences(HSA_FENCE_SCOPE_NONE, variant.scope));
            break;
          case Kind::KernelAcquire:
            dispatch(queues[1],
                     variant.scope == HSA_FENCE_SCOPE_SYSTEM ? hsa.acquireSystem : hsa.acquireAgent,
                     tinyArgs,
                     none,
                     kernelFences);
            break;
          case Kind::KernelRelease:
            dispatch(queues[1],
                     variant.scope == HSA_FENCE_SCOPE_SYSTEM ? hsa.releaseSystem : hsa.releaseAgent,
                     tinyArgs,
                     none,
                     kernelFences);
            break;
          case Kind::KernelNothing:
            dispatch(queues[1], hsa.nothing, tinyArgs, none, kernelFences);
            break;
          default:
            break;
          }
          ++enqueued;
        }
        barrierAnd(
            queues[1], none, otherDone, fences(HSA_FENCE_SCOPE_NONE, HSA_FENCE_SCOPE_SYSTEM));
      }
      const auto t0 = std::chrono::steady_clock::now();
      hsa_signal_store_screlease(gate, 0);
      waitBelowHost(victimDone, 1);
      const auto t1 = std::chrono::steady_clock::now();
      // how far the other queue got meanwhile
      const auto read = hsa_queue_load_read_index_scacquire(queues[1]);
      static std::uint64_t base = 0;
      const double fencesPerKernel = variant.kind == Kind::None ? 0.0
                                                                : static_cast<double>(read - base) /
                                                                      static_cast<double>(kernels);
      if (variant.kind != Kind::None) {
        waitBelowHost(otherDone, 1);
        base = hsa_queue_load_write_index_relaxed(queues[1]);
      }
      static_cast<void>(enqueued);
      std::printf("  %8.2f (%6.1f)", us(t0, t1) / kernels, fencesPerKernel);
      std::fflush(stdout);
    }
    std::printf("\n");
  }
  return 0;
}
