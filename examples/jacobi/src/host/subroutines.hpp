#ifndef JSOLVER_SUBROUTINES_HPP
#define JSOLVER_SUBROUTINES_HPP

#include "datatypes.hpp"
#include <functional>
#include <tuple>

namespace host {
void multMatVec(const CpuMatrixDataT &matrix, const VectorT &v, VectorT &res);

void manipVectors(const RangeT &range, const VectorT &a, const VectorT &b, VectorT &res,
                  std::function<real(real, real)> &&ops);

real getInfNorm(const RangeT &range, const VectorT &vector);
} // namespace host
#endif // JSOLVER_SUBROUTINES_HPP
