# Intel MPI + SYCL example

## Build
Create a build folder and run `cmake .. -DDEVICE_BACKEND=ONEAPI -DREAL_SIZE_IN_BYTES=8 -DDEVICE_SUB_ARCH=dg1`.

## Run
Make sure to allow GPU aware MPI by setting `export I_MPI_OFFLOAD=2`.
Then run the executable `mpirun -n 2 ./mpi`.