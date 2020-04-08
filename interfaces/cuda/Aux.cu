#include <iostream>
#include <string>
#include <cuda.h>

#include "CudaWrappedAPI.h"
#include "Internals.h"

using namespace device;

void ConcreteAPI::compareDataWithHost(const real *HostPtr,
                                      const real *DevPtr,
                                      const size_t NumElements,
                                      const std::string& DataName) {

  std::cout << "DEVICE:: comparing array: " << DataName << '\n';

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
  std::cout << "DEVICE:: host and device arrays are the same\n";
  delete [] Temp;
};


__global__ void kernel_checkOffloading() {
  printf("gpu offloading is working\n");
}


void ConcreteAPI::checkOffloading() {
  kernel_checkOffloading<<<1,1>>>(); CHECK_ERR;
  cudaDeviceSynchronize();
}


__global__ void kernel_scaleArray(real *Array, const real Scalar, const size_t NumElements) {
  unsigned Index = threadIdx.x + blockIdx.x * blockDim.x;
  if (Index < NumElements) {
    Array[Index] *= Scalar;
  }
}

void ConcreteAPI::scaleArray(real *DevArray, const real Scalar, const size_t NumElements) {
  dim3 Block(32, 1, 1);
  dim3 Grid = internals::computeGrid1D(Block,  NumElements);
  kernel_scaleArray<<<Grid, Block>>>(DevArray, Scalar, NumElements); CHECK_ERR;
}
