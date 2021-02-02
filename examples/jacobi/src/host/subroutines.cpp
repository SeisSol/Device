#include "subroutines.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace host {
void multMatVec(const CpuMatrixDataT &matrix, const VectorT &v, VectorT &res) {
  const auto &info = matrix.info;
  const RangeT &range = info.range;

  assert(static_cast<size_t>(info.numRows) == v.size() && "matrix and vec. must have the same size");

  std::fill(&res[range.start], &res[range.end], 0.0);

  for (int columnIdx = 0; columnIdx < info.maxNonZerosPerRow; ++columnIdx) {
    for (int row = range.start; row != range.end; ++row) {
      int linIndex = row + columnIdx * info.numRows;
      int column = matrix.indices[linIndex];
      if (column != NIN)
        res[row] += (matrix.data[linIndex] * v[column]);
    }
  }
}

void manipVectors(const RangeT &range, const VectorT &a, const VectorT &b, VectorT &res,
                  std::function<real(real, real)> &&ops) {
  assert(a.size() == b.size() && "Vectors A and B must have the same size");
  assert(a.size() == res.size() && "Vectors A, B and Res must have the same size");

  for (int i = range.start; i != range.end; ++i) {
    res[i] = ops(a[i], b[i]);
  }
}

real getInfNorm(const RangeT &range, const VectorT &vector) {
  using TypeT = VectorT::iterator::value_type;

  const auto* maxElemItr =
      std::max_element(&vector[range.start],
                       &vector[range.end],
                       [](const TypeT &a, const TypeT &b) { return std::abs(a) < std::abs(b); });

  return std::abs(*maxElemItr);
}
} // namespace host