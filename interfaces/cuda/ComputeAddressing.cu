#include "AbstractAPI.h"


namespace device {
  namespace addressing {
    // in case of a base array and offsets
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