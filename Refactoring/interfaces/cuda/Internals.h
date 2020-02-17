#ifndef DEVICE_INTERNALS_H
#define DEVICE_INTERNALS_H

#include <string>

#define CHECK_ERR device::internals::checkErr(__FILE__,__LINE__)
namespace device {
  namespace internals {
    void checkErr(const std::string &file, int line);
    inline dim3 computeGrid1D(const dim3 &Block, const size_t Size) {
      int NumBlocks = (Size + Block.x - 1) / Block.x;
      return dim3(NumBlocks, 1, 1);
    }
  }
}

#endif //DEVICE_INTERNALS_H
