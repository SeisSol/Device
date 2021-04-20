#include "helper.hpp"
#include "host/subroutines.hpp"
#include "matrix_manip.hpp"
#include "solvers.hpp"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <limits>

#include "kernels/subroutinesGPU.h"
#include <device.h>
using namespace device;

void gpu::Solver::run(const SolverSettingsT &settings, const CpuMatrixDataT &matrix, const VectorT &inputRhs,
                      VectorT &x) {

  // allocate all necessary data structs
  const WorkSpaceT &ws = matrix.info.ws;
  const RangeT range = matrix.info.range;
  VectorAssembler assembler(ws, range);

  VectorT residual(matrix.info.numRows, std::numeric_limits<real>::max());
  VectorT temp(matrix.info.numRows, 0.0);
  real infNorm = std::numeric_limits<real>::max();
  unsigned currentIter{0};

  // assume that RHS is distributed. Thus, let's assemble it
  VectorT rhs(inputRhs.size(), 0.0);
  assembler.assemble(const_cast<real *>(inputRhs.data()), const_cast<real *>(rhs.data()));

  // compute diag and LU matrices
  VectorT invDiag;
  CpuMatrixDataT lu(MatrixInfoT(WorkSpaceT{}, matrix.info.numRows, 0));
  std::tie(invDiag, lu) = getDLU(matrix);

  // allocate gpu data structures
  DeviceInstance &device = DeviceInstance::getInstance();
  device.api->setDevice(ws.rank);
  this->setUp(lu, rhs, x, residual, invDiag);

  // start solver
  Statistics computeStat(ws, range);
  Statistics commStat(ws, range);
  while ((infNorm > settings.eps) and (currentIter <= settings.maxNumIters)) {

    computeStat.start();
    launch_multMatVec(*devLU, devX, devTemp);
    launch_manipVectors(range, devRhs, devTemp, devX, VectorManipOps::Subtraction);
    launch_manipVectors(range, devInvDiag, devX, devX, VectorManipOps::Multiply);
    device.api->synchDevice();
    computeStat.stop();

    commStat.start();
    assembler.assemble<SystemType::OnDevice>(devX, devTempX);
    commStat.stop();
    std::swap(devX, devTempX);

    // Compute residual and print output
    if ((currentIter % settings.printInfoNumIters) == 0) {

      launch_manipVectors(range, devDiag, devX, devTemp, VectorManipOps::Multiply);
      launch_multMatVec(*devLU, devX, devResidual);
      launch_manipVectors(range, devTemp, devResidual, devResidual, VectorManipOps::Addition);
      launch_manipVectors(range, devRhs, devResidual, devResidual, VectorManipOps::Subtraction);
      device.api->copyFrom(const_cast<real *>(residual.data()), devResidual, residual.size() * sizeof(real));

      auto localInfNorm = infNorm = host::getInfNorm(range, residual);
#ifdef USE_MPI
      MPI_Allreduce(&localInfNorm, &infNorm, 1, MPI_CUSTOM_REAL, MPI_MAX, ws.comm);
#else
      infNorm = localInfNorm;
#endif
      std::stringstream stream;
      stream << "Current iter: " << currentIter << "; Residual: " << infNorm;
      if (currentIter != 0) {
        stream << "; compute: " << computeStat.getStatistics().mean << ' ' << Statistics::getUnits();
#ifdef USE_MPI
        stream << "; comm: " << commStat.getStatistics().mean << ' ' << Statistics::getUnits();
#endif
      }
      Logger(ws, 0) << stream;
    }
    ++currentIter;
  }

  device.api->synchDevice();
  assembler.assemble<SystemType::OnDevice>(devX, devTempX);
  device.api->copyFrom(const_cast<real *>(x.data()), devTempX, x.size() * sizeof(real));

  this->tearDown();
}

void gpu::Solver::setUp(const CpuMatrixDataT &lu, const VectorT &rhs, const VectorT &x, const VectorT &residual,
                        VectorT &invDiag) {
  DeviceInstance &device = DeviceInstance::getInstance();

  devRhs = static_cast<real *>(device.api->allocGlobMem(rhs.size() * sizeof(real)));
  device.api->copyTo(devRhs, rhs.data(), rhs.size() * sizeof(real));

  devX = static_cast<real *>(device.api->allocGlobMem(x.size() * sizeof(real)));
  device.api->copyTo(devX, x.data(), x.size() * sizeof(real));

  devTempX = static_cast<real *>(device.api->allocGlobMem(x.size() * sizeof(real)));
  device.api->copyTo(devTempX, x.data(), x.size() * sizeof(real));

  devTemp = static_cast<real *>(device.api->allocGlobMem(x.size() * sizeof(real)));

  // InvDiag still holds the diagonal elements at this point
  devDiag = static_cast<real *>(device.api->allocGlobMem(invDiag.size() * sizeof(real)));
  device.api->copyTo(devDiag, invDiag.data(), invDiag.size() * sizeof(real));

  // compute inverse diagonal matrix
  std::transform(invDiag.begin(), invDiag.end(), invDiag.begin(), [](const real &diag) {
    assert(diag != 0.0 && "diag element cannot be equal to zero");
    return 1.0 / diag;
  });

  devInvDiag = static_cast<real *>(device.api->allocGlobMem(invDiag.size() * sizeof(real)));
  device.api->copyTo(devInvDiag, invDiag.data(), invDiag.size() * sizeof(real));

  devResidual = static_cast<real *>(device.api->allocGlobMem(residual.size() * sizeof(real)));
  device.api->copyTo(devResidual, residual.data(), residual.size() * sizeof(real));

  devLU = std::make_unique<GpuMatrixDataT>(lu.info);
  // GpuMatrixDataT devLU(LU.Info);
  devLU->data = static_cast<real *>(device.api->allocGlobMem(lu.info.volume * sizeof(real)));
  device.api->copyTo(devLU->data, lu.data.data(), lu.info.volume * sizeof(real));

  devLU->indices = static_cast<int *>(device.api->allocGlobMem(lu.info.volume * sizeof(int)));
  device.api->copyTo(devLU->indices, lu.indices.data(), lu.info.volume * sizeof(int));
}

void gpu::Solver::tearDown() {
  DeviceInstance &device = DeviceInstance::getInstance();
  device.api->freeMem(devResidual);
  device.api->freeMem(devInvDiag);
  device.api->freeMem(devDiag);
  device.api->freeMem(devTemp);
  device.api->freeMem(devX);
  device.api->freeMem(devTempX);
  device.api->freeMem(devRhs);

  device.api->freeMem(devLU->data);
  device.api->freeMem(devLU->indices);
  devLU.reset(nullptr);
}
