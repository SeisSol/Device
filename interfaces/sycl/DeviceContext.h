// SPDX-FileCopyrightText: 2022 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_SYCL_DEVICECONTEXT_H_
#define SEISSOLDEVICE_INTERFACES_SYCL_DEVICECONTEXT_H_

#include "DeviceQueues.h"
#include "Statistics.h"

#include <unordered_map>

namespace device {

/*
 * Container class holding the queue, stack, and buffer of the current device
 */
class DeviceContext {
  public:
  explicit DeviceContext(const sycl::device& targetDevice);
  std::unordered_map<void*, size_t> memoryToSizeMap;
  DeviceQueues queueBuffer;
  Statistics statistics;

  private:
  void onExceptionOccurred(sycl::exception_list& exceptions);
};
} // namespace device

#endif // SEISSOLDEVICE_INTERFACES_SYCL_DEVICECONTEXT_H_
