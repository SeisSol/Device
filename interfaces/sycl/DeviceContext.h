// SPDX-FileCopyrightText: 2022-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_SYCL_DEVICECONTEXT_H_
#define SEISSOLDEVICE_INTERFACES_SYCL_DEVICECONTEXT_H_

#include "DeviceCircularQueueBuffer.h"
#include "Statistics.h"
#include <unordered_map>

namespace device {

/*
 * Container class holding the queue, stack, and buffer of the current device
 */
class DeviceContext {
public:
  DeviceContext(const cl::sycl::device &targetDevice, size_t concurrencyLevel);
  std::unordered_map<void *, size_t> memoryToSizeMap;
  DeviceCircularQueueBuffer queueBuffer;
  Statistics statistics;
private:
  void onExceptionOccurred(cl::sycl::exception_list &exceptions);
};
} // namespace device

#endif // SEISSOLDEVICE_INTERFACES_SYCL_DEVICECONTEXT_H_

