#ifndef JSOLVER_MATRIX_SUBROUTINES_HPP
#define JSOLVER_MATRIX_SUBROUTINES_HPP

#include "datatypes.hpp"
#include <tuple>

CpuMatrixDataT init2DStencilMatrix(WorkSpaceT ws, int numRows);
std::pair<VectorT, CpuMatrixDataT> getDLU(const CpuMatrixDataT& matrix);

#endif //JSOLVER_MATRIX_SUBROUTINES_HPP
