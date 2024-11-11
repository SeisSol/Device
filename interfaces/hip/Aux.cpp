// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "HipWrappedAPI.h"
#include "Internals.h"

namespace device {
__global__ void kernel_checkOffloading() {
  printf("gpu offloading is working\n");
}

void ConcreteAPI::checkOffloading() {
  hipLaunchKernelGGL(kernel_checkOffloading, dim3(1), dim3(1), 0, 0);
  CHECK_ERR;
  hipDeviceSynchronize();
}
} // namespace device

