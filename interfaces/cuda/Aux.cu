#include "CudaWrappedAPI.h"
#include "Internals.h"

namespace device {
__global__ void kernel_checkOffloading() { printf("gpu offloading is working\n"); }

void ConcreteAPI::checkOffloading() {
  isFlagSet<StatusId::DeviceSelected>();
  kernel_checkOffloading<<<1, 1>>>();
  CHECK_ERR;
  cudaDeviceSynchronize();
}
} // namespace device