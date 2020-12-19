# Parallel Jacobi Iteration Benchmark

This is MPI+GPU Parallel Jacobi Iteration Solver where a GPU part can be either CUDA or HIP.

Tthe benchmark is hard-coded with 2D stencil matrix of the form [-1 2 -1], 
however, it can be extended if it is needed.  The program starts with computing 
the right-hand side by multiplying the matrix with a vector of ones (a.k.a a fake RHS).
The solver starts with a random vector **x** and expected to converge to the solution: 
a vector of ones.

The matrix is stored in the sparse ELLPACK form.

### Dependencies
Make sure that you have the following packages installed in your system:

- OpenMPI
- Yaml-cpp

These are only needed if you want to run the tests:

- GTest
- GMock


If you are using a Nvidia GPU, make sure to install CUDA.

If you are using an AMD GPU, make sure to install HIP.


### Compile without MPI
```
# either
cmake .. -DWITH_MPI=OFF -DDEVICE_BACKEND=CUDA -DSM=6=sm_60
# or
cmake .. -DWITH_MPI=OFF -DDEVICE_BACKEND=HIP
```

### Support of different floating point precision formats
```
# either double 
cmake .. -DWITH_MPI=OFF -DDEVICE_BACKEND=HIP -DREAL_SIZE_IN_BYTES=8
# or single
cmake .. -DWITH_MPI=OFF -DDEVICE_BACKEND=HIP -DREAL_SIZE_IN_BYTES=4
```

## MPI Support
```
#either double 
cmake .. -DWITH_MPI=ON -DDEVICE_BACKEND=HIP -DREAL_SIZE_IN_BYTES=4
```


#### How to run
Compile the project and run as following:
```console
mpirun -n <num_proc> solver <intput>.yaml
```
*input/example.yaml* - is an example that you can use and modify.

Edit the following in the *input file* if you want to run the program with CPU-only: 
```console
sovler_type: "cpu"
```

To use GPUs:
```console
sovler_type: "gpu"
```

#### How to test
Run tests as following:
```console
mpirun -n 3 ./tests
``` 
NOTE: Do not use more than 3 MPI processes for testing. 
