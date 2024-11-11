// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "CudaWrappedAPI.h"
#include "Internals.h"

namespace device {
__global__ void kernel_checkOffloading() { printf("gpu offloading is working\n"); }

void ConcreteAPI::checkOffloading() {
  isFlagSet<StatusID::DeviceSelected>(status);
  kernel_checkOffloading<<<1, 1>>>();
  CHECK_ERR;
  cudaDeviceSynchronize();
}
} // namespace device

