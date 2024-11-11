// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "AbstractAPI.h"
#include "interfaces/hip/Internals.h"
#include "utils/logger.h"
#include <cassert>
#include <device.h>

namespace device {

void Algorithms::compareDataWithHost(const real *hostPtr, const real *devPtr, const size_t numElements,
                                     const std::string &dataName) {

  std::stringstream stream;
  stream << "DEVICE:: comparing array: " << dataName << '\n';

  real *temp = new real[numElements];
  hipMemcpy(temp, devPtr, numElements * sizeof(real), hipMemcpyDeviceToHost);
  CHECK_ERR;
  constexpr real EPS = 1e-12;
  for (unsigned i = 0; i < numElements; ++i) {
    if (abs(hostPtr[i] - temp[i]) > EPS) {
      if ((std::isnan(hostPtr[i])) || (std::isnan(temp[i]))) {
        stream << "DEVICE:: results is NAN. Cannot proceed\n";
        logError() << stream.str();
      }

      stream << "DEVICE::ERROR:: host and device arrays are different\n";
      stream << "DEVICE::ERROR:: "
             << "host value (" << hostPtr[i] << ") | "
             << "device value (" << temp[i] << ") "
             << "at index " << i << '\n';
      stream << "DEVICE::ERROR:: Difference = " << (hostPtr[i] - temp[i]) << std::endl;
      delete[] temp;
      logError() << stream.str();
    }
  }
  stream << "DEVICE:: host and device arrays are the same\n";
  logInfo() << stream.str();
  delete[] temp;
};
} // namespace device

