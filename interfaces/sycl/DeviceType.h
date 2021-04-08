#ifndef DEVICE_TYPE_H
#define DEVICE_TYPE_H

#include <CL/sycl.hpp>
#include <string>

using namespace cl::sycl;

namespace device {
enum class DeviceType { GPU = 0, CPU = 1, FPGA = 2, HOST = 3, OTHERS = 4 };

std::string convertToString(cl::sycl::info::device_type type);
DeviceType fromSyclType(cl::sycl::info::device_type type);
bool compare(cl::sycl::info::device_type typeA, cl::sycl::info::device_type typeB);

} // namespace device

#endif