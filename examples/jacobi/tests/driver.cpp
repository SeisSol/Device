#include <gtest/gtest.h>
#include <mpi.h>
#include "datatypes.hpp"
#include <stdexcept>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  WorkSpaceT ws{MPI_COMM_WORLD};
  if (ws.size > 3)
    throw std::runtime_error("Tests were ran with more than 3 MPI processes");

  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}