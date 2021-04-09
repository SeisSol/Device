#include <CL/sycl.hpp>
#include <device.h>
#include <mpi.h>
#include <stdio.h>

using namespace sycl;
using namespace device;

void fork_other(int otherRank) {

  DeviceInstance &device = DeviceInstance::getInstance();
  auto *api = device.api;
  api->setDevice(otherRank);

  auto *dev_ptr =  (int *) api->allocGlobMem(sizeof(int));
  const int value = 42;
  api->copyTo(dev_ptr, &value, sizeof(int));

  printf("sending value to other GPU\n");
  MPI_Send(dev_ptr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  printf("successfully sent to GPU\n");
}

void fork_root(int rootRank) {
  DeviceInstance &device = DeviceInstance::getInstance();
  auto *api = device.api;
  api->setDevice(rootRank);

  auto *dev_ptr =  (int *) api->allocGlobMem(sizeof(int));
  int value = -1;
  api->copyTo(dev_ptr, &value, sizeof(int));

  printf("waiting for value from GPU\n");
  MPI_Recv(dev_ptr, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  api->copyFrom(&value, dev_ptr, sizeof(int));
  printf("value from GPU received: %d\n", value);
}

// ToDo: apply api libs; currently disabled as minimal example
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    fork_root(world_rank);
  } else if (world_rank == 1) {
    fork_other(world_rank);
  }

  MPI_Finalize();
}