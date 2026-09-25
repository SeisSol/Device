// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "algorithms/Instantiations.h"
#include "utils/logger.h"

#include <cmath>
#include <device.h>
#include <sstream>
#include <string>

namespace device {

template <typename T>
void Algorithms::compareDataWithHost(const T* hostPtr,
                                     const T* devPtr,
                                     const size_t numElements,
                                     const std::string& dataName) {

  std::stringstream stream;
  stream << "DEVICE:: comparing array: " << dataName << '\n';

  constexpr T Eps = 1e-12;
  for (size_t i = 0; i < numElements; ++i) {
    if (std::abs(hostPtr[i] - devPtr[i]) > Eps) {
      if ((std::isnan(hostPtr[i])) || (std::isnan(devPtr[i]))) {
        stream << "DEVICE:: results is NAN. Cannot proceed\n";
        logError() << stream.str();
      }

      stream << "DEVICE::ERROR:: host and device arrays are different\n";
      stream << "DEVICE::ERROR:: "
             << "host value (" << hostPtr[i] << ") | "
             << "device value (" << devPtr[i] << ") "
             << "at index " << i << '\n';
      stream << "DEVICE::ERROR:: Difference = " << (hostPtr[i] - devPtr[i]) << std::endl;
      logError() << stream.str();
    }
  }
  stream << "DEVICE:: host and device arrays are the same\n";
  logInfo() << stream.str();
}

DEVICE_ALGORITHMS_FLOATING_TYPES(DEVICE_ALGORITHMS_INSTANTIATE_COMPARE_DATA_WITH_HOST)

} // namespace device
