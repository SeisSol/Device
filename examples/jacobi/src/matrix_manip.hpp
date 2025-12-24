// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_EXAMPLES_JACOBI_SRC_MATRIX_MANIP_HPP_
#define SEISSOLDEVICE_EXAMPLES_JACOBI_SRC_MATRIX_MANIP_HPP_

#include "datatypes.hpp"

#include <tuple>

CpuMatrixDataT init2DStencilMatrix(WorkSpaceT ws, int numRows);
std::pair<VectorT, CpuMatrixDataT> getDLU(const CpuMatrixDataT& matrix);

#endif // SEISSOLDEVICE_EXAMPLES_JACOBI_SRC_MATRIX_MANIP_HPP_
