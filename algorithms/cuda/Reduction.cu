#include "AbstractAPI.h"
#include "../../interfaces/cuda/Internals.h"
#include <device.h>
#include <cassert>

namespace device {
  inline size_t getNearestPow2Number(size_t number) {
    auto power = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(number))));
    constexpr size_t BASE = 1;
    return (BASE << power);
  }

  template<typename T>
  struct Sum {
    __device__ T getDefaultValue() {
      return static_cast<T>(0);
    }
    __device__ T operator()(T op1, T op2) {
      return op1 + op2;
    }
  };

  template<typename T>
  struct Subtract {
    __device__ T getDefaultValue() {
      return static_cast<T>(0);
    }
    __device__ T operator()(T op1, T op2) {
      return op1 - op2;
    }
  };


  template<typename T, typename OperationT>
  void __global__ kernel_reduce(T *to, const T *from, const size_t size, OperationT Operation) {
    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    T value = (id < size) ? from[id] : Operation.getDefaultValue();

    constexpr unsigned int fullMask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
      value = Operation(value, __shfl_down_sync(fullMask, value, offset));
    }

    if (threadIdx.x == 0) {
      to[blockIdx.x] = value;
    }
  }

  template<typename T>
  T Algorithms::reduceVector(T* buffer, size_t size, const ReductionType type) {
    assert(api != nullptr && "api has not been attached to algorithms sub-system");
    size_t adjustedSize = device::getNearestPow2Number(size);
    const size_t totalBuffersSize = 2 * adjustedSize * sizeof(T);

    T *reductionBuffer = reinterpret_cast<T*>(api->getStackMemory(totalBuffersSize));

    this->fillArray(reinterpret_cast<char*>(reductionBuffer),
                    static_cast<char>(0),
                    2 * adjustedSize * sizeof(T)); CHECK_ERR;
    cudaMemcpy(reductionBuffer, buffer, size * sizeof(T), cudaMemcpyDeviceToDevice);

    T* buffer0 = &reductionBuffer[0];
    T* buffer1 = &reductionBuffer[adjustedSize];

    dim3 block(internals::WARP_SIZE, 1, 1);
    dim3 grid = internals::computeGrid1D(internals::WARP_SIZE, size);

    size_t swapCounter = 0;
    for (size_t reducedSize = adjustedSize; reducedSize > 0; reducedSize /= internals::WARP_SIZE) {
      switch(type) {
        case ReductionType::Add: {
          kernel_reduce<<<grid, block>>>(buffer1, buffer0, reducedSize, device::Sum<T>());
          break;
        }
        case ReductionType::Subtract: {
          kernel_reduce<<<grid, block>>>(buffer1, buffer0, reducedSize, device::Subtract<T>());
          break;
        }
        default : {
          assert(false && "reduction type is not implemented");
        }
        CHECK_ERR;
      }
      std::swap(buffer1, buffer0);
      ++swapCounter;
    }

    T results{};
    if ((swapCounter % 2) == 0) {
      cudaMemcpy(&results, buffer0, sizeof(T), cudaMemcpyDeviceToHost);
    }
    else {
      cudaMemcpy(&results, buffer1, sizeof(T), cudaMemcpyDeviceToHost);
    }
    CHECK_ERR;
    this->fillArray(reinterpret_cast<char*>(reductionBuffer),
                    static_cast<char>(0),
                    2 * adjustedSize * sizeof(T)); CHECK_ERR;
    api->popStackMemory();
    return results;
  }

  template int Algorithms::reduceVector(int* buffer, size_t size, ReductionType type);
  template real Algorithms::reduceVector(real* buffer, size_t size, ReductionType type);
} // namespace device