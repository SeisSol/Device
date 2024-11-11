// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HIP_STATISTICS_H_
#define SEISSOLDEVICE_INTERFACES_HIP_STATISTICS_H_

namespace device {
  struct Statistics {
    size_t allocatedMemBytes = 0;
    size_t allocatedUnifiedMemBytes = 0;
    size_t deallocatedMemBytes = 0;

    size_t explicitlyTransferredDataToDeviceBytes = 0;
    size_t explicitlyTransferredDataToHostBytes = 0;
  };
}


#endif // SEISSOLDEVICE_INTERFACES_HIP_STATISTICS_H_

