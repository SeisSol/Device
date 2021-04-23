#include <device.h>
#include <iostream>
#include <mpi.h>

using namespace device;

void forkOther(int otherRank) {

  DeviceInstance &device = DeviceInstance::getInstance();
  auto *api = device.api;
  api->setDevice(otherRank);

  auto *devPtr = (int *)api->allocGlobMem(sizeof(int));
  const int value = 42;
  api->copyTo(devPtr, &value, sizeof(int));

  std::cout << "sending value to other GPU\n";
  MPI_Send(devPtr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  std::cout << "successfully sent to GPU\n";
}

void forkRoot(int rootRank) {
  DeviceInstance &device = DeviceInstance::getInstance();
  auto *api = device.api;
  api->setDevice(rootRank);

  auto *devPtr = (int *)api->allocGlobMem(sizeof(int));
  int value = -1;
  api->copyTo(devPtr, &value, sizeof(int));

  std::cout << "waiting for value from GPU\n";
  MPI_Recv(devPtr, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  api->copyFrom(&value, devPtr, sizeof(int));
  std::cout << "value from GPU received: " << value << '\n';
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int worldRank{}, mpiSize{};
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

  if (mpiSize == 2) {
    if (worldRank == 0) {
      forkRoot(worldRank);
    } else if (worldRank == 1) {
      forkOther(worldRank);
    }
  }
  else {
    if (worldRank == 0) {
      std::cerr << "error: ran with more or less than 2 MPI processes\n";
    }
  }

  MPI_Finalize();
}