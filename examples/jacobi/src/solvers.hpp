#ifndef JSOLVER_SOLVERS_HPP
#define JSOLVER_SOLVERS_HPP

#include "datatypes.hpp"
#include <memory>

enum SolverType { Cpu, Gpu };

struct SolverSettingsT {
  real eps{1e-6};
  unsigned maxNumIters{100};
  unsigned printInfoNumIters{10};
  SolverType solverType{SolverType::Cpu};
};

namespace host {
void solver(const SolverSettingsT &settings, const CpuMatrixDataT &matrix, const VectorT &rhs, VectorT &guess);
} // namespace host

namespace gpu {
class Solver {
public:
  void run(const SolverSettingsT &settings, const CpuMatrixDataT &matrix, const VectorT &rhs, VectorT &guess);

private:
  void setUp(const CpuMatrixDataT &lu, const VectorT &rhs, const VectorT &x, const VectorT &residual, VectorT &invDiag);

  void tearDown();

  std::unique_ptr<GpuMatrixDataT> devLU{};
  real *devRhs{};
  real *devX{};
  real *devTempX{};
  real *devTemp{};
  real *devDiag{};
  real *devInvDiag{};
  real *devResidual{};
};
} // namespace gpu

#endif // JSOLVER_SOLVERS_HPP
