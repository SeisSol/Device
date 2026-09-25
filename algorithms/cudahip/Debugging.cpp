// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "AbstractAPI.h"
#include "Internals.h"
#include "algorithms/Instantiations.h"
#include "utils/logger.h"

#include <cassert>
#include <device.h>

namespace device {

template <typename T>
void Algorithms::compareDataWithHost(const T* hostPtr,
                                     const T* devPtr,
                                     const size_t numElements,
                                     const std::string& dataName) {

  std::stringstream stream;
  stream << "DEVICE:: comparing array: " << dataName << '\n';

  T* temp = new T[numElements];
  api->copyFrom(temp, devPtr, numElements * sizeof(T));
  CHECK_ERR;
  constexpr T EPS = 1e-12;
  for (size_t i = 0; i < numElements; ++i) {
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

DEVICE_ALGORITHMS_FLOATING_TYPES(DEVICE_ALGORITHMS_INSTANTIATE_COMPARE_DATA_WITH_HOST)

} // namespace device
