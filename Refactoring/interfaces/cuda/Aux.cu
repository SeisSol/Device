#include <iostream>
#include <string>
#include <cuda.h>

#include "CudaInterface.h"
#include "Internals.h"

using namespace device;

void ConcreteInterface::compareDataWithHost(const real *HostPtr,
                                            const real *DevPtr,
                                            const size_t NumElements,
                                            const char *ArrayName) {
  if (ArrayName) {
    std::cout << "DEVICE:: comparing array: " << ArrayName << "\n";
  }

  real* Temp = new real[NumElements];
  cudaMemcpy(Temp, DevPtr, NumElements * sizeof(real), cudaMemcpyDeviceToHost); CHECK_ERR;
  const real EPS = 1e-12;
  for (unsigned i = 0; i < NumElements; ++i) {
    if (abs(HostPtr[i] - Temp[i]) > EPS) {
      if ((std::isnan(HostPtr[i])) || (std::isnan(Temp[i]))) {
        std::cout << "DEVICE:: results is NAN. Cannot proceed\n";
        throw;
      }

      std::cout << "DEVICE::ERROR:: host and device arrays are different\n";
      std::cout << "DEVICE::ERROR:: "
                << "host value (" << HostPtr[i] << ") | "
                << "device value (" << Temp[i] << ") "
                << "at index " << i
                << '\n';
      std::cout << "DEVICE::ERROR:: Difference = " << (HostPtr[i] - Temp[i]) << std::endl;
      delete [] Temp;
      throw;
    }
  }
  std::cout << "DEVICE: host and device arrays are the same\n";
  delete [] Temp;
};


__global__ void kernel_checkOffloading() {
  printf("gpu offloading is working\n");
}


void ConcreteInterface::checkOffloading() {
  kernel_checkOffloading<<<1,1>>>(); CHECK_ERR;
  cudaDeviceSynchronize();
}