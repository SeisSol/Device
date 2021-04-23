#ifndef JSOLVER_SUBROUTINESGPU_HPP
#define JSOLVER_SUBROUTINESGPU_HPP

#include "../../datatypes.hpp"
#include "../../simpleDatatypes.hpp"

enum VectorManipOps { Addition, Subtraction, Multiply };

void launch_multMatVec(const GpuMatrixDataT &matrix, const real *v, real *res, void* streamPtr);
void launch_manipVectors(const RangeT &range,
                         const real *a,
                         const real *b,
                         real *res,
                         VectorManipOps op,
                         void* streamPtr);

#endif // JSOLVER_SUBROUTINESGPU_HPP
