#ifndef JSOLVER_DATATYPES_HPP
#define JSOLVER_DATATYPES_HPP

#include <cassert>
#include <limits>
#include <mpi.h>
#include <stdio.h>
#include <vector>

#include "simpleDatatypes.hpp"

#if REAL_SIZE == 8
#define MPI_CUSTOM_REAL MPI_DOUBLE
#else
#define MPI_CUSTOM_REAL MPI_FLOAT
#endif

using VectorT = std::vector<real>;

struct WorkSpaceT {
  WorkSpaceT(MPI_Comm comm = MPI_COMM_WORLD) : comm(comm) {
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
      isFirst = true;
    if (rank == (size - 1))
      isLast = true;
  }
  WorkSpaceT(const WorkSpaceT &other) = default;

  int rank{std::numeric_limits<int>::infinity()};
  int size{std::numeric_limits<int>::infinity()};
  bool isFirst{false};
  bool isLast{false};
  MPI_Comm comm;
};

struct MatrixInfoT {
  MatrixInfoT(WorkSpaceT ws, int rows, int nonZerosPerRow) : ws(ws), numRows{rows}, maxNonZerosPerRow{nonZerosPerRow} {

    // distribute data
    int chunk = numRows / ws.size;

    range.start = ws.rank * chunk;
    range.end = ws.isLast ? rows : (ws.rank + 1) * chunk;
    localNumRows = range.end - range.start;

    assert(localNumRows > 1 && "matrix should contain enough elements for each MPI process");
    volume = numRows * maxNonZerosPerRow;
    localVolume = localNumRows * maxNonZerosPerRow;
  }
  MatrixInfoT(const MatrixInfoT &other) = default;

  int numRows{};
  int localNumRows{};
  int volume{};
  int localVolume{};
  int maxNonZerosPerRow{};
  RangeT range{};
  WorkSpaceT ws;
};

struct CpuMatrixDataT {
  explicit CpuMatrixDataT(const MatrixInfoT &info) : info(info) {
    data.resize(info.volume, 0.0);
    indices.resize(info.volume, 0.0);
  }

  CpuMatrixDataT(const CpuMatrixDataT &other) = default;
  ~CpuMatrixDataT() = default;

  std::vector<real> data{};
  std::vector<int> indices{};
  MatrixInfoT info;
};

struct GpuMatrixDataT {
  explicit GpuMatrixDataT(const MatrixInfoT &info) : info(info) {}

  real *data{};
  int *indices{};
  MatrixInfoT info;
};

#endif // JSOLVER_DATATYPES_HPP
