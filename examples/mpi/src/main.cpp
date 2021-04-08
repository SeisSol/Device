#include <CL/sycl.hpp>
#include <mpi.h>
#include <stdio.h>

using namespace sycl;

void fork_other(int otherRank) {
    queue q{gpu_selector{}};
    // it breaks due to the device malloc
    auto *dev_ptr = malloc_device<int>(1, q);
    const int value = 42;
    auto e = q.memcpy(dev_ptr, &value, 1 * sizeof(int));
    e.wait();

    printf("sending value to CPU from GPU\n");
    MPI_Send(dev_ptr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    printf("successfully sent to CPU\n");
}

void fork_root(int rootRank) {
    queue q{cpu_selector{}};
    int ptr = -1;
    printf("waiting for value from GPU\n");
    MPI_Recv(&ptr, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("value from GPU received: %d\n", ptr);
}

//ToDo: apply api libs; currently disabled as minimal example
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