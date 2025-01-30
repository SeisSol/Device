// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "matrix_manip.hpp"
#include <algorithm>
#include <cassert>

CpuMatrixDataT init2DStencilMatrix(WorkSpaceT ws, int numRows) {
  MatrixInfoT matrixInfo(ws, numRows, 3);
  CpuMatrixDataT matrix(matrixInfo);

  for (int columnIdx = 0; columnIdx < matrixInfo.maxNonZerosPerRow; ++columnIdx) {
    for (int row = 0; row < matrixInfo.numRows; ++row) {
      int linIndex = row + columnIdx * matrixInfo.numRows;
      matrix.indices[linIndex] = row + columnIdx - 1;
      matrix.data[linIndex] = (matrix.indices[linIndex] == row) ? 2.0 : -1.0;

      if ((matrix.indices[linIndex] < 0) or (matrix.indices[linIndex]) >= matrix.info.numRows) {
        matrix.indices[linIndex] = NIN;
        matrix.data[linIndex] = 0.0;
      }
    }
  }

  return matrix;
}

std::pair<VectorT, CpuMatrixDataT> getDLU(const CpuMatrixDataT &matrix) {
  const auto &info = matrix.info;
  VectorT diagonal(info.numRows, std::numeric_limits<real>::infinity());
  CpuMatrixDataT lu(MatrixInfoT(info.ws, info.numRows, info.maxNonZerosPerRow - 1));

  std::vector<int> columnCounters(info.numRows, 0);
  for (int columnIdx = 0; columnIdx < info.maxNonZerosPerRow; ++columnIdx) {
    for (int row = 0; row < info.numRows; ++row) {
      int origLinIndex = row + columnIdx * info.numRows;
      if (matrix.indices[origLinIndex] == row) {
        diagonal[row] = matrix.data[origLinIndex];
      } else {
        int luLinIndex = row + columnCounters[row] * info.numRows;
        lu.data[luLinIndex] = matrix.data[origLinIndex];
        lu.indices[luLinIndex] = matrix.indices[origLinIndex];
        ++columnCounters[row];
      }
    }
  }
  return std::make_pair(diagonal, lu);
}

