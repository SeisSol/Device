// SPDX-FileCopyrightText: 2021 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_SYCL_DEVICETYPE_H_
#define SEISSOLDEVICE_INTERFACES_SYCL_DEVICETYPE_H_

#include <string>
#include <sycl/sycl.hpp>

namespace device {
enum class DeviceType { GPU = 0, CPU = 1, FPGA = 2, HOST = 3, OTHERS = 4 };

std::string convertToString(sycl::info::device_type type);
DeviceType fromSyclType(sycl::info::device_type type);
bool compare(sycl::device devA, sycl::device devB);

} // namespace device

#endif // SEISSOLDEVICE_INTERFACES_SYCL_DEVICETYPE_H_
