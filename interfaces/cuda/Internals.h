#ifndef DEVICE_INTERNALS_H
#define DEVICE_INTERNALS_H

#include <string>

#define CHECK_ERR device::internals::checkErr(__FILE__, __LINE__)
namespace device {
namespace internals {
void checkErr(const std::string &file, int line);
inline dim3 computeGrid1D(const dim3 &block, const size_t size) {
  int numBlocks = (size + block.x - 1) / block.x;
  return dim3(numBlocks, 1, 1);
}

inline dim3 computeGrid1D(const int &leadingDim, const size_t size) {
  int numBlocks = (size + leadingDim - 1) / leadingDim;
  return dim3(numBlocks, 1, 1);
}

inline dim3 computeBlock1D(const int &leadingDim, const size_t size) {
  int numItems = ((size + leadingDim - 1) / leadingDim) * leadingDim;
  return dim3(numItems, 1, 1);
}

constexpr int WARP_SIZE = 32;
} // namespace internals
} // namespace device

#endif // DEVICE_INTERNALS_H
