#ifndef DEVICE_COMPUTE_ADDRESSING_H
#define DEVICE_COMPUTE_ADDRESSING_H

namespace device {
  namespace addressing {
    __device__ real* findData(real *Data, unsigned Stride, unsigned BlockId) {
      return &Data[BlockId * Stride];
    }


    __device__ const real* findData(const real *Data, unsigned Stride, unsigned BlockId) {
      return &Data[BlockId * Stride];
    }


    __device__ real* findData(real** Data, unsigned Stride, unsigned BlockId) {
      return &(Data[BlockId][Stride]);
    }


    __device__ const real* findData(const real** Data, unsigned Stride, unsigned BlockId) {
      return &(Data[BlockId][Stride]);
    }
  }
}
#endif //DEVICE_COMPUTE_ADDRESSING_H
