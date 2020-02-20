#include "common.h"
#include <cuda_runtime.h>
#include <cuda.h>

namespace device {
  namespace query {

    void init() {
      cuInit(0); CUDA_CHECK;
    }

    int getNumDevices() {
      int numDevices{};
      cudaGetDeviceCount(&numDevices); CUDA_CHECK;
      return numDevices;
    }

    void setDevice(int device_id) {
      cudaSetDevice(device_id); CUDA_CHECK;
    }
  }
}