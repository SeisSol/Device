// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Arch.h"

#include <array>
#include <string>
#include <string_view>

namespace device::hsaqueues {

namespace {
struct Entry {
  // the name, or its beginning where `prefix`
  std::string_view name;
  bool prefix;
  ArchTraits::Tested tested;
  bool barrierValue;
  bool spreadPipes;
  bool deviceKernargs;
  std::string_view note;
};

using Tested = ArchTraits::Tested;

// (the first entry that matches counts)
constexpr std::array Table{
    Entry{"gfx90a",
          false,
          Tested::Runs,
          true,
          true,
          true,
          "MI200 (MI250X): the command processor serves the queues of a process on 4 pipes, which "
          "they share (spread: TPV13 on 2 nodes 1.92 s instead of 2.43 s); kernel arguments in "
          "device memory: TPV13 1.7 % less"},
    Entry{
        "gfx942",
        false,
        Tested::Runs,
        true,
        false,
        true,
        "MI300 (MI300A): no shared pipes found; kernel arguments in device memory: TPV5 and TPV13 "
        "9-10 % less on four ranks"},
    Entry{"gfx94",
          true,
          Tested::No,
          true,
          false,
          true,
          "MI300 (gfx940, gfx941): as gfx942, untested"},
    Entry{"gfx950",
          false,
          Tested::No,
          true,
          false,
          false,
          "MI350: barrier-value packets as CLR uses them there, untested"},
    Entry{
        "gfx1150",
        false,
        Tested::UnitTest,
        false,
        false,
        false,
        "Radeon 890M: the command processor rejects barrier-value packets (INVALID_PACKET_FORMAT); "
        "queues wait by barrier-AND packets on a signal of each mark, as HIP does"},
    // (before the other gfx12: gfx1250, gfx1251)
    Entry{"gfx125",
          true,
          Tested::No,
          false,
          false,
          false,
          "gfx12.5: CLR leaves barrier-value packets off there, keeps kernel arguments in device "
          "memory, and writes kernels as extended dispatch packets (with a dependent signal); "
          "untested whether the command processor takes standard dispatch packets"},
    Entry{"gfx11",
          true,
          Tested::No,
          false,
          false,
          false,
          "RDNA 3: no barrier-value packets (as on the 890M); untested"},
    Entry{"gfx12",
          true,
          Tested::No,
          false,
          false,
          false,
          "RDNA 4: CLR leaves barrier-value packets off there; untested"},
};
} // namespace

ArchTraits archTraits(std::string_view name) {
  // without the features (e.g. gfx90a:sramecc+:xnack-)
  const auto plain = name.substr(0, name.find(':'));
  ArchTraits traits;
  traits.name = std::string(plain);
  for (const auto& entry : Table) {
    if (plain == entry.name || (entry.prefix && plain.substr(0, entry.name.size()) == entry.name)) {
      traits.known = true;
      traits.tested = entry.tested;
      traits.barrierValue = entry.barrierValue;
      traits.spreadPipes = entry.spreadPipes;
      traits.deviceKernargs = entry.deviceKernargs;
      traits.note = std::string(entry.note);
      return traits;
    }
  }
  traits.note = "not in the table of HSA queues (hsaqueues/Arch.cpp): the cautious defaults";
  return traits;
}

} // namespace device::hsaqueues
