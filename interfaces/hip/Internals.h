#ifndef DEVICE_INTERNALS_H
#define DEVICE_INTERNALS_H

#include "hip/hip_runtime.h"
#include <string>

#define CHECK_ERR device::internals::checkErr(__FILE__,__LINE__)
namespace device {
  namespace internals {
    void checkErr(const std::string &file, int line);
    inline dim3 computeGrid1D(const dim3 &Block, const size_t Size) {
      int NumBlocks = (Size + Block.x - 1) / Block.x;
      return dim3(NumBlocks, 1, 1);
    }

    inline dim3 computeGrid1D(const int &LeadingDim, const size_t Size) {
      int NumBlocks = (Size + LeadingDim - 1) / LeadingDim;
      return dim3(NumBlocks, 1, 1);
    }

    inline dim3 computeBlock1D(const int &LeadingDim, const size_t Size) {
      int NumItems = ((Size + LeadingDim - 1) / LeadingDim) * LeadingDim;
      return dim3(NumItems, 1, 1);
    }

    constexpr int WARP_SIZE = 64;
  }
}

#endif //DEVICE_INTERNALS_H
