// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "AbstractAPI.h"
#include "interfaces/cuda/Internals.h"
#include "utils/logger.h"
#include <cassert>
#include <device.h>

namespace device {

template<typename T>
void Algorithms::compareDataWithHost(const T *hostPtr, const T *devPtr, const size_t numElements,
                                     const std::string &dataName) {

  std::stringstream stream;
  stream << "DEVICE:: comparing array: " << dataName << '\n';

  T *temp = new T[numElements];
  cudaMemcpy(temp, devPtr, numElements * sizeof(T), cudaMemcpyDeviceToHost);
  CHECK_ERR;
  constexpr T EPS = 1e-12;
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

template void Algorithms::compareDataWithHost(const float *hostPtr, const float *devPtr, const size_t numElements,
  const std::string &dataName);
  template void Algorithms::compareDataWithHost(const double *hostPtr, const double *devPtr, const size_t numElements,
    const std::string &dataName);

} // namespace device

