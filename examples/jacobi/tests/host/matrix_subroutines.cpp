// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "datatypes.hpp"
#include "host/subroutines.hpp"
#include "matrix_manip.hpp"
#include "helper.hpp"
#ifdef USE_MPI
#include <mpi.h>
#endif

using ::testing::ElementsAreArray;

TEST(Matrix, MatrixMult) {
  WorkSpaceT ws{MPI_COMM_WORLD};
  CpuMatrixDataT matrix = init2DStencilMatrix(ws, 7);
  VectorT vector{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  VectorT test(7, 0);
  VectorT tmp(7, 0);
  host::multMatVec(matrix, vector, test);

  VectorAssembler assembler(ws, matrix.info.range);
  assembler.assemble(test.data(), tmp.data());

  VectorT resMustBe{-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0};
  ASSERT_THAT(tmp, ElementsAreArray(resMustBe));
}

