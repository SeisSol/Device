// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HIP_OWNQUEUES_COMMON_H_
#define SEISSOLDEVICE_INTERFACES_HIP_OWNQUEUES_COMMON_H_

#include <amd_comgr/amd_comgr.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hsa/amd_hsa_queue.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <string>
#include <vector>

namespace device::ownqueues {

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

/// the seconds of steady_clock
double steadySeconds();
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

inline std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

/// stops with an error unless the call of HSA succeeded
void check(hsa_status_t status, const char* call);
void checkComgr(amd_comgr_status_t status, const char* call);
/// a warning, once per message
void warnOnce(const std::string& message);

/// the device can access the host memory: its address there, otherwise nullptr
const void* deviceAddress(const void* ptr);

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

} // namespace device::ownqueues

#endif // SEISSOLDEVICE_INTERFACES_HIP_OWNQUEUES_COMMON_H_
