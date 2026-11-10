// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Kernels.h"

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace device::hsaqueues {

// ------------------------------------------------------------------------------------------------
// kernels of our own: copies, fills, and waits where the command processor cannot wait

__global__ void hsaCopy(char* dst, const char* src, std::size_t bytes) {
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
__global__ void hsaFill(char* dst,
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

// the waits below read with the scope of the system: the values come from the command processor or
// the host, past the caches of the device

// until the value of a signal is below `below`
__global__ void hsaWaitBelow(const std::int64_t* value, std::int64_t below) {
  while (__hip_atomic_load(value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) >= below) {
    __builtin_amdgcn_s_sleep(2);
  }
}

// until the value of a signal is at least `atLeast`
__global__ void hsaWaitAtLeast(const std::int64_t* value, std::int64_t atLeast) {
  while (__hip_atomic_load(value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < atLeast) {
    __builtin_amdgcn_s_sleep(2);
  }
}

// takes the time (by its completion signal, on a queue with profiling)
__global__ void hsaStamp() {}

// keeps a compute unit busy for `ticks` of the constant clock (see samePipe())
__global__ void hsaSpin(std::uint64_t ticks) {
  const auto start = wall_clock64();
  while (static_cast<std::uint64_t>(wall_clock64() - start) < ticks) {
    __builtin_amdgcn_s_sleep(1);
  }
}

// until a counter in host memory is at least `atLeast`
__global__ void hsaWaitHost(const std::uint32_t* counter, std::uint32_t atLeast) {
  while (__hip_atomic_load(counter, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < atLeast) {
    __builtin_amdgcn_s_sleep(2);
  }
}

// on a stream of HIP, which cannot wait for a signal of ours: until the counter of the marks of an
// HSA queue is below `below` (loads of the system read past the L2 caches, which the command
// processor does not update)
__global__ void hsaHipWaitBelow(const std::int64_t* value, std::int64_t below) {
  while (__hip_atomic_load(value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) >= below) {
    __builtin_amdgcn_s_sleep(2);
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

// on a stream of HIP, after the work before it there: sets a counter that HSA queues wait for (see
// recordEvent())
__global__ void hsaWriteCounter(std::int64_t* counter, std::int64_t value) {
  __hip_atomic_store(counter, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// lets the work after it see what the host and other agents wrote before, together with the
// acquire fence of the agent of the next kernel (for the caches of the other compute units): a
// fence of the system in the shader drops only the lines of the L2 cache that may be stale (on
// gfx90a, those of memory that is not local; the hardware keeps those of local memory coherent),
// while that of a packet drops all of them, at the expense of the kernels of all queues (on the
// MI250X, a kernel that read 4 MiB again and again took 3.3 times as long next to a queue of such
// packets, and 1.1 times next to one of such kernels)
__global__ void hsaAcquireSystem() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""); }

} // namespace device::hsaqueues
