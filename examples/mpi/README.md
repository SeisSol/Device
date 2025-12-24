<!--
    SPDX-FileCopyrightText: 2019 SeisSol Group

    SPDX-License-Identifier: BSD-3-Clause
-->

# Intel MPI + SYCL example

## Build

Create a build folder and run
`cmake .. -DDEVICE_BACKEND=oneapi -DREAL_SIZE_IN_BYTES=8 -DDEVICE_ARCH=dg1`.
Note that Intel GPU aware MPI does only work when using Intel low level platform.
The sycl device backend automatically implements this rule.

## Run

Make sure to allow GPU aware MPI by setting `export I_MPI_OFFLOAD=2`.
Then run the executable `mpirun -n 2 ./mpi`.
