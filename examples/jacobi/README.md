# Parallel Jacobi Solver

This Parallel Jacobi Solver is based on the MPI Parallel Jacobi Solver and can be compiled for GPU usage.

Currently, the solver is hard-coded with 2D stencil matrix of the form [-1 2 -1], 
however, it can be extended if it is needed.  The program starts with computing 
the right-hand side by multiplying the matrix with a vector of ones (a.k.a a fake RHS).
The solver starts with a random vector **x** and expected to converge to the solution: 
a vector of ones.

The matrix is stored in the sparse ELLPACK form (see, docs).

### Dependencies
Make sure that you have the following packages installed in your system:

- OpenMPI
- Yaml-cpp

These are only needed if you want to run the tests:

- GTest
- GMock

## GPU Parallel Jacobi Solver

This is a modificated MPI Parallel Jacobi Solver using a GPU to calculate the single iterations of X.
You can compile it for AMD GPUs using HIP or Nvidia using CUDA.

The used GPU is defined via
```console
set(DEVICE_BACKEND "HIP" CACHE STRING "type of an interface")
```

in CMakeLists.txt.

Similar you can adjust if float or double gets used for the computation
```console
set(REAL_SIZE_IN_BYTES "8" CACHE STRING "size of the floating point data type")
```

### Dependencies

If you are using a Nvidia GPU, make sure to install CUDA.
If you are using an AMD GPU, make sure to install HIP.

#### How to run
Compile the project and run as following:
```console
mpirun -n <num_proc> solver <intput>.yaml
```
*input/example.yaml* - is an example that you can use and modify.

If you want to run the program using the CPU put
```console
sovler_type: "cpu"
```
in your input file; for GPU use
```console
sovler_type: "gpu"
```

#### How to test
Do not use more than 3 MPI processes for testing. You can run tests as following:
```console
mpirun -n 3 ./tests
``` 

## Minimal MPI Settings
#### OpenMPI
```console
# for CPU:
export OMPI_MCA_btl=vader,self

# for GPU: make sure you have OpenMPI cinfigured with UCX
``` 
