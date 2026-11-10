// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_KERNELS_H_
#define SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_KERNELS_H_

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace device::hsaqueues {

// kernels of our own: copies, fills, and waits where the command processor cannot wait (see
// Kernels.cpp); the queues launch them by their kernel objects, as any other kernel
__global__ void hsaCopy(char* dst, const char* src, std::size_t bytes);
__global__ void hsaFill(char* dst,
                        std::uint32_t value,
                        std::uint32_t elementSize,
                        std::size_t width,
                        std::size_t height,
                        std::size_t pitch);
__global__ void hsaWaitBelow(const std::int64_t* value, std::int64_t below);
__global__ void hsaWaitAtLeast(const std::int64_t* value, std::int64_t atLeast);
__global__ void hsaStamp();
__global__ void hsaSpin(std::uint64_t ticks);
__global__ void hsaWaitHost(const std::uint32_t* counter, std::uint32_t atLeast);
__global__ void hsaHipWaitBelow(const std::int64_t* value, std::int64_t below);
__global__ void hsaWriteCounter(std::int64_t* counter, std::int64_t value);
__global__ void hsaAcquireSystem();

} // namespace device::hsaqueues

#endif // SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_KERNELS_H_
