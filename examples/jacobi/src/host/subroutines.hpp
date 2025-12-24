// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_EXAMPLES_JACOBI_SRC_HOST_SUBROUTINES_HPP_
#define SEISSOLDEVICE_EXAMPLES_JACOBI_SRC_HOST_SUBROUTINES_HPP_

#include "datatypes.hpp"

#include <functional>
#include <tuple>

namespace host {
void multMatVec(const CpuMatrixDataT& matrix, const VectorT& v, VectorT& res);

void manipVectors(const RangeT& range,
                  const VectorT& a,
                  const VectorT& b,
                  VectorT& res,
                  std::function<real(real, real)>&& ops);

real getInfNorm(const RangeT& range, const VectorT& vector);
} // namespace host

#endif // SEISSOLDEVICE_EXAMPLES_JACOBI_SRC_HOST_SUBROUTINES_HPP_
