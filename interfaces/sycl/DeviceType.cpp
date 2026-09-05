// SPDX-FileCopyrightText: 2021 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DeviceType.h"

#include "utils/env.h"

#include <iostream>
#include <string>
#include <utility>

namespace device {

DeviceType fromSyclType(sycl::info::device_type type) {
  if (type == sycl::info::device_type::gpu) {
    return DeviceType::GPU;
  }
  if (type == sycl::info::device_type::cpu) {
    return DeviceType::CPU;
  }
  if (type == sycl::info::device_type::host) {
    return DeviceType::HOST;
  }
  if (type == sycl::info::device_type::accelerator) {
    return DeviceType::FPGA;
  }
  return DeviceType::OTHERS;
}

bool compare(const sycl::device& devA, const sycl::device& devB) {
  std::string preferred;
  preferred += utils::Env("").get("PREFERRED_DEVICE_TYPE", "");

  // A device of the preferred type sorts ahead of every other device, and the rest follow the
  // order of the DeviceType enum. Deciding the two directions independently - as in "A wins if it
  // matches, B wins if it matches" - makes both compare(a, b) and compare(b, a) true for two
  // devices of the preferred type, and sorting on such a comparator is undefined.
  const auto rank = [&preferred](const sycl::device& device) {
    const auto type = device.get_info<sycl::info::device::device_type>();
    const auto matchesPreference = convertToString(type) == preferred ? 0 : 1;
    return std::make_pair(matchesPreference, static_cast<int>(fromSyclType(type)));
  };

  return rank(devA) < rank(devB);
}

std::string convertToString(sycl::info::device_type type) {
  switch (fromSyclType(type)) {
  case DeviceType::GPU:
    return "GPU";
  case DeviceType::CPU:
    return "CPU";
  case DeviceType::HOST:
    return "HOST";
  case DeviceType::FPGA:
    return "FPGA";
  default:
    return "OTHERS";
  }
}
} // namespace device
