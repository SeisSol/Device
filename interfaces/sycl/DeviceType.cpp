#include "DeviceType.h"
#include "utils/env.h"
#include <iostream>

namespace device {

DeviceType fromSyclType(cl::sycl::info::device_type type) {
  if (type == cl::sycl::info::device_type::gpu) {
    return DeviceType::GPU;
  }
  if (type == cl::sycl::info::device_type::cpu) {
    return DeviceType::CPU;
  }
  if (type == cl::sycl::info::device_type::host) {
    return DeviceType::HOST;
  }
  if (type == cl::sycl::info::device_type::accelerator) {
    return DeviceType::FPGA;
  }
  return DeviceType::OTHERS;
}

bool compare(cl::sycl::info::device_type typeA, cl::sycl::info::device_type typeB) {
  std::string env;
  env += utils::Env::get("PREFERRED_DEVICE_TYPE", "");

  if (convertToString(typeA).compare(env) == 0) {
    return true;
  }
  if (convertToString(typeB).compare(env) == 0) {
    return false;
  }
  return fromSyclType(typeA) < fromSyclType(typeB);
}

std::string convertToString(cl::sycl::info::device_type type) {
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