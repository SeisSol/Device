// SPDX-FileCopyrightText: 2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_SYCL_STATISTICS_H_
#define SEISSOLDEVICE_INTERFACES_SYCL_STATISTICS_H_

#include <sstream>

namespace device {
struct Statistics {
  size_t allocatedMemBytes = 0;
  size_t allocatedUnifiedMemBytes = 0;
  size_t deallocatedMemBytes = 0;

  size_t explicitlyTransferredDataToDeviceBytes = 0;
  size_t explicitlyTransferredDataToHostBytes = 0;

  std::string getSummary() {
    std::ostringstream report{};
    report << "Allocated Memory (bytes): " << allocatedMemBytes << "\n";
    report << "of it Unified Memory (bytes): " << allocatedUnifiedMemBytes << "\n";
    report << "Deallocated Memory (bytes): " << deallocatedMemBytes << "\n";
    report << "Bytes transferred to device memory: " << explicitlyTransferredDataToDeviceBytes << "\n";
    report << "Bytes transferred to host memory: " << explicitlyTransferredDataToHostBytes << "\n";
    return report.str();
  }
};
} // namespace device


#endif // SEISSOLDEVICE_INTERFACES_SYCL_STATISTICS_H_

