#include "AbstractAPI.h"
#include "../../interfaces/cuda/Internals.h"
#include <device.h>
#include <cassert>

namespace device {
  template<typename T>
  __global__ void kernel_scaleArray(T *array, const T scalar, const size_t numElements) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < numElements) {
      array[index] *= scalar;
    }
  }

  template<typename T>
  void Algorithms::scaleArray(T *devArray, T scalar, const size_t numElements) {
    dim3 block(64, 1, 1);
    dim3 grid = internals::computeGrid1D(block,  numElements);
    kernel_scaleArray<<<grid, block>>>(devArray, scalar, numElements); CHECK_ERR;
  }
  template void Algorithms::scaleArray(real *devArray, real scalar, const size_t numElements);
  template void Algorithms::scaleArray(int *devArray, int scalar, const size_t numElements);
  template void Algorithms::scaleArray(char *devArray, char scalar, const size_t numElements);


//--------------------------------------------------------------------------------------------------
  template<typename T>
  __global__ void kernel_fillArray(T *array, T scalar, const size_t numElements) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < numElements) {
      array[index] = scalar;
    }
  }

  template<typename T>
  void Algorithms::fillArray(T *devArray, const T scalar, const size_t numElements) {
    dim3 block(64, 1, 1);
    dim3 grid = internals::computeGrid1D(block,  numElements);
    kernel_scaleArray<<<grid, block>>>(devArray, scalar, numElements); CHECK_ERR;
  }
  template void Algorithms::fillArray(real *devArray, real scalar, const size_t numElements);
  template void Algorithms::fillArray(int *devArray, int scalar, const size_t numElements);
  template void Algorithms::fillArray(char *devArray, char scalar, const size_t numElements);


//--------------------------------------------------------------------------------------------------
  __global__ void kernel_touchMemory(real *ptr, size_t size, bool clean) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size) {
      if (clean) {
        ptr[id] = 0;
      }
      else {
        real value = ptr[id];
        // Do something dummy here. We just need to check the pointers point to valid memory locations.
        // Avoid compiler optimization. Possibly, implement a dummy code with asm.
        value += 1;
        value -= 1;
      }
    }
  }

  void Algorithms::touchMemory(real *ptr, size_t size, bool clean) {
    dim3 block(256, 1, 1);
    dim3 grid = internals::computeGrid1D(block, size);
    kernel_touchMemory<<<grid, block>>>(ptr, size, clean); CHECK_ERR;
  }
} // namespace device