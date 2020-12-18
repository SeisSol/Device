#ifndef JSOLVER_HELPER_HPP
#define JSOLVER_HELPER_HPP

#include "datatypes.hpp"
#include <cstring>
#include <iostream>
#include <limits>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <ostream>
#include <sstream>
#include <string>

#define NOT_A_RANK -1

class Logger {
public:
  explicit Logger(const WorkSpaceT &ws, int rank = NOT_A_RANK) : ws{ws} {
    this->rank = (rank == NOT_A_RANK) ? ws.rank : rank;
  }
  void operator<<(const std::ostream &stream) {
    waitPrevious();
    if (ws.rank == rank) {
      std::cout << std::string(80, '=') << std::endl;
      std::cout << "Info from Rank: " << ws.rank << std::endl;
      std::cout << stream.rdbuf() << std::endl;
    }
    callNext();
#ifdef USE_MPI
    MPI_Barrier(ws.comm);
#endif
  }

private:
  void waitPrevious() {
#ifdef USE_MPI
    if (!ws.isFirst) {
      MPI_Recv(&dummyData, 1, MPI_INT, ws.rank - 1, 0, ws.comm, &status);
    }
#endif
  }

  void callNext() {
#ifdef USE_MPI
    if (!ws.isLast) {
      MPI_Send(&dummyData, 1, MPI_INT, ws.rank + 1, 0, ws.comm);
    }
#endif
  }
  int rank{NOT_A_RANK};
  const WorkSpaceT &ws{};
  int dummyData{};
#ifdef USE_MPI
  MPI_Status status;
#endif
};

class VectorAssembler {
public:
  VectorAssembler(WorkSpaceT ws, RangeT range) : ws(ws) {

    recvCounts.resize(ws.size, 0);
    displs.resize(ws.size, 0);

    recvCounts[ws.rank] = range.end - range.start;
    displs[ws.rank] = range.start;
#ifdef USE_MPI
    MPI_Allgather(&recvCounts[ws.rank], 1, MPI_INT, recvCounts.data(), 1, MPI_INT, ws.comm);
    MPI_Allgather(&displs[ws.rank], 1, MPI_INT, displs.data(), 1, MPI_INT, ws.comm);
#endif
  }

  void assemble(real *src, real *dest) {
#ifdef USE_MPI
    MPI_Allgatherv(&src[displs[ws.rank]], recvCounts[ws.rank], MPI_CUSTOM_REAL, (void *)dest, recvCounts.data(),
                   displs.data(), MPI_CUSTOM_REAL, ws.comm);
#else
    if (src != dest) {
      std::memcpy(reinterpret_cast<char*>(dest), reinterpret_cast<char*>(src), recvCounts[0] * sizeof(real));
    }
#endif
  }

private:
  WorkSpaceT ws;
  std::vector<int> recvCounts;
  std::vector<int> displs;
};

#endif // JSOLVER_HELPER_HPP
