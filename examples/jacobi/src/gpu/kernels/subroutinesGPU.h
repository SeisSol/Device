// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_EXAMPLES_JACOBI_SRC_GPU_KERNELS_SUBROUTINESGPU_H_
#define SEISSOLDEVICE_EXAMPLES_JACOBI_SRC_GPU_KERNELS_SUBROUTINESGPU_H_

#include "../../datatypes.hpp"
#include "../../simpleDatatypes.hpp"

enum VectorManipOps { Addition, Subtraction, Multiply };

void launch_multMatVec(const GpuMatrixDataT& matrix, const real* v, real* res, void* streamPtr);
void launch_manipVectors(const RangeT& range,
                         const real* a,
                         const real* b,
                         real* res,
                         VectorManipOps op,
                         void* streamPtr);

#endif // SEISSOLDEVICE_EXAMPLES_JACOBI_SRC_GPU_KERNELS_SUBROUTINESGPU_H_
