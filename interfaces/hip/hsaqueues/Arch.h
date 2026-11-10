// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_ARCH_H_
#define SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_ARCH_H_

#include <cstdint>
#include <string>
#include <string_view>

namespace device::hsaqueues {

/**
 * What HSA queues know of an architecture of AMD GPUs: what its command processor takes, and the
 * defaults that measurements there chose (see the table in Arch.cpp). An architecture that the
 * table lacks gets the cautious defaults: waits by barrier-AND packets on a signal of each mark
 * (standard packets, as HIP waits), kernel arguments in host memory, queues anywhere. The
 * environment decides otherwise in any case (DEVICE_HSA_QUEUE_WAITS, _PIPES and _KERNARGS).
 *
 * HSA queues only get enabled on architectures marked Runs, unless DEVICE_HSA_QUEUE_UNTESTED=1.
 * For a new architecture: run the test of HSA queues and its benchmarks there (barrier-value
 * packets, the pipes that queues share, the costs of fences), add a line to the table, and mark it
 * Runs once programs ran there with HSA queues, bitwise as without. An entry may hold for the APUs
 * or the discrete GPUs of an architecture only (e.g. MI300A and MI300X, both gfx942); what the
 * command processor does also depends on its firmware, whose version the log tells.
 */
struct ArchTraits {
  enum class Tested : std::uint8_t {
    // nothing ran there
    No,
    // the test of HSA queues ran there
    UnitTest,
    // programs ran there with HSA queues, bitwise as without
    Runs
  };

  /// the name of the architecture as the agent tells it (e.g. gfx90a)
  std::string name;
  /// whether the device is an APU (one memory with the host, e.g. MI300A), as the agent tells it
  bool apu{false};
  /// whether the table has it; otherwise, the cautious defaults
  bool known{false};
  Tested tested{Tested::No};
  /// whether the command processor waits for signal values (barrier-value packets), instead of a
  /// kernel that waits
  bool barrierValue{false};
  /// whether HSA queues get spread over the pipes of the command processor that the queues of a
  /// process share (see Runtime::placeQueue()), instead of going anywhere
  bool spreadPipes{false};
  /// whether the kernel arguments go into fine-grained device memory by default, instead of host
  /// memory
  bool deviceKernargs{false};
  /// whether kernels go as extended dispatch packets (with a dependent signal), as CLR writes them
  /// there (gfx12.5), instead of kernel dispatch packets
  bool extDispatch{false};
  /// the devices, and what the table entry rests on (for the log)
  std::string note;
};

/// the traits of the architecture `name` (e.g. gfx90a, or gfx90a:sramecc+:xnack-), of an APU or not
ArchTraits archTraits(std::string_view name, bool apu);

} // namespace device::hsaqueues

#endif // SEISSOLDEVICE_INTERFACES_HIP_HSAQUEUES_ARCH_H_
