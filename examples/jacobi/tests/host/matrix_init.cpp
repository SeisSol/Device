// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "datatypes.hpp"
#include "matrix_manip.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

using ::testing::ElementsAreArray;

TEST(Matrix, MatrixInit) {
  WorkSpaceT ws{MPI_COMM_WORLD};
  std::vector<real> refData{0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.0,  2.0,  2.0, 2.0,
                            2.0, 2.0,  2.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0};

  std::vector<int> refIndices{NIN, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, NIN};

  CpuMatrixDataT hostMatrix = init2DStencilMatrix(ws, 7);
  ASSERT_THAT(hostMatrix.data, ElementsAreArray(refData));
  ASSERT_THAT(hostMatrix.indices, ElementsAreArray(refIndices));
}

TEST(Matrix, DaingAndLU) {
  WorkSpaceT ws{MPI_COMM_WORLD};
  std::vector<real> refLU{
      0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0};

  std::vector<int> refLUIndices{NIN, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, NIN};

  VectorT refDiag = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  CpuMatrixDataT origMatrix = init2DStencilMatrix(ws, 7);

  VectorT testDiag;
  CpuMatrixDataT testLU(MatrixInfoT(ws, 7, 0));
  std::tie(testDiag, testLU) = getDLU(origMatrix);

  ASSERT_THAT(testDiag, ElementsAreArray(refDiag));
  ASSERT_THAT(testLU.data, ElementsAreArray(refLU));
  ASSERT_THAT(testLU.indices, ElementsAreArray(refLUIndices));
}
