// SPDX-FileCopyrightText: 2022 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DeviceContext.h"

#include "DeviceType.h"
#include "utils/logger.h"

namespace device {
DeviceContext::DeviceContext(const sycl::device& targetDevice, size_t concurrencyLevel)
    : queueBuffer{DeviceCircularQueueBuffer{
          targetDevice, [&](sycl::exception_list l) { onExceptionOccurred(l); }, concurrencyLevel}},
      statistics{Statistics{}} {}

void DeviceContext::onExceptionOccurred(sycl::exception_list& exceptions) {
  for (auto& excep : exceptions) {
    try {
      rethrow_exception(excep);
    } catch (const sycl::exception& exc) {
      logError() << "SYCL API error:" << std::string(exc.what());
    } catch (const std::exception& exc) {
      logError() << "SYCL API/C++ error:" << std::string(exc.what());
    }
  }
}
} // namespace device
