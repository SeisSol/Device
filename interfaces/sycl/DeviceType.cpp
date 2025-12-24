// SPDX-FileCopyrightText: 2021 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DeviceType.h"

#include "utils/env.h"

#include <iostream>

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

bool compare(sycl::device devA, sycl::device devB) {
  std::string env;
  env += utils::Env("").get("PREFERRED_DEVICE_TYPE", "");

  auto typeA = devA.get_info<sycl::info::device::device_type>();
  auto typeB = devB.get_info<sycl::info::device::device_type>();

  if (convertToString(typeA).compare(env) == 0) {
    return true;
  }
  if (convertToString(typeB).compare(env) == 0) {
    return false;
  }

  // devices of same type are sorted by their cl::deviceid
  if (typeA == typeB) {
    // return devA.get() < devB.get();
  }

  return fromSyclType(typeA) < fromSyclType(typeB);
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
