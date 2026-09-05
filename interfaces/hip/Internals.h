// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HIP_INTERNALS_H_
#define SEISSOLDEVICE_INTERFACES_HIP_INTERNALS_H_

#include "hip/hip_runtime.h"

#include <initializer_list>

#define APIWRAP(call) (void)::device::internals::checkResult(call, __FILE__, __LINE__, {})
#define APIWRAPX(call, except) ::device::internals::checkResult(call, __FILE__, __LINE__, except)
#define CHECK_ERR APIWRAP(hipGetLastError())

namespace device::internals {

constexpr static int DefaultBlockDim = 1024;

using DeviceStreamT = hipStream_t;
// Every wrapped call goes through here, so the parameters stay free of anything that allocates:
// the file name is the string literal __FILE__ expands to, and the accepted errors are read from
// the caller's temporary array.
hipError_t checkResult(hipError_t error,
                       const char* file,
                       int line,
                       std::initializer_list<hipError_t> except);
inline dim3 computeGrid1D(const dim3& block, const size_t size) {
  int numBlocks = (size + block.x - 1) / block.x;
  return dim3(numBlocks, 1, 1);
}

inline dim3 computeGrid1D(const int& leadingDim, const size_t size) {
  int numBlocks = (size + leadingDim - 1) / leadingDim;
  return dim3(numBlocks, 1, 1);
}

inline dim3 computeBlock1D(const int& leadingDim, const size_t size) {
  int numItems = ((size + leadingDim - 1) / leadingDim) * leadingDim;
  return dim3(numItems, 1, 1);
}

} // namespace device::internals

#endif // SEISSOLDEVICE_INTERFACES_HIP_INTERNALS_H_
