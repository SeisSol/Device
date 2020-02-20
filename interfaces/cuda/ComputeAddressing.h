#ifndef DEVICE_COMPUTE_ADDRESSING_H
#define DEVICE_COMPUTE_ADDRESSING_H

// TODO: add namespace

// in case of a base array and offsets
namespace device {
  namespace addressing {
    __device__ real* findData(real *Data, unsigned Stride, unsigned BlockId);
    __device__ const real* findData(const real *Data, unsigned Stride, unsigned BlockId);
    __device__ real* findData(real** Data, unsigned Stride, unsigned BlockId);
    __device__ const real* findData(const real** Data, unsigned Stride, unsigned BlockId);
  }
}
#endif //DEVICE_COMPUTE_ADDRESSING_H
