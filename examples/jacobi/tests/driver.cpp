#include <gtest/gtest.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "datatypes.hpp"
#include <stdexcept>

int main(int argc, char **argv) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  WorkSpaceT ws{MPI_COMM_WORLD};
  if (ws.size > 3)
    throw std::runtime_error("Tests were ran with more than 3 MPI processes");
#else
  WorkSpaceT ws{MPI_COMM_WORLD};
#endif

  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
#ifdef USE_MPI
  MPI_Finalize();
#endif
  return result;
}