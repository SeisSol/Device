# GPU API wrapper
This repository wraps a GPU (or basically any other accelerator) API.

## Structure
The folder structure sets up like this:
* algorithms: Implements common parallel algorithms for the API
* cmake: contains specific cmake files if necessary
* examples: contains a basic example and a jacobi relaxation example    
* interfaces: implements the API for a specific vendor
* submodules: git submodules  needed for this repository; resolve with `git submodule init` and `git submodule update`
* tests: gtests for the algorithms package

## Current implementations
Currently, there are three implementations available
* Nvidia CUDA
* AMD HIP
* SYCL implemented by Intel OneAPI and AdaptiveCpp (https://github.com/AdaptiveCpp/AdaptiveCpp.git)

## Environment Variables

* `DEVICE_STACK_MEM_SIZE` - the size (in GB) of the device memory to be allocated for the stack (`1` GB by default)
* `DEVICE_MAX_CONCURRENCY_LEVEL` - the num. streams/queues to be allocated for the circular stream buffer (`4` by
  default)

## Setup and Build
The setup is explained with linux commands; for windows, see the belonging batch reference.

* Create a folder: `mkdir -p build`
* Switch into the folder: `cd build`
* Now setup CMake with `cmake .. <DEVICE_OPTIONS> -DREAL_SIZE_IN_BYTES=<PRECISION>`; 
* The real size defines the precision level of the floating point operations; use a precision of 4 or 8 bytes
* The device options are explained in the following

If you want to run the examples, follow the instructions in the belonging package.

### Device options for CUDA
* use `-DDEVICE_BACKEND:STRING=cuda` to build the CUDA implementation 
* CUDA also requires a sub architecture for the device, for example `-DDEVICE_ARCH=sm60`; see the CMakeLists.txt for all options
* Make sure to have CUDA including nvcc installed 
* Complete example call: `cmake .. -DDEVICE_BACKEND:STRING=cuda -DREAL_SIZE_IN_BYTES=4 -DDEVICE_ARCH=sm60`

### Device options for HIP
* use `-DDEVICE_BACKEND:STRING=hip` to build the HIP implementation 
* Complete example call: `cmake .. -DDEVICE_BACKEND:STRING=hip -DREAL_SIZE_IN_BYTES=4 -DDEVICE_ARCH=gfx906`

### Device options for OneAPI
* use `-DDEVICE_BACKEND:STRING=oneapi` to build the OneAPI implementation
* currently, OneAPI is mainly used to target Intel devices; for CUDA or HIP as device backend, use Open SYCL and set the right environment variables, respectively
* Set the environment variable `PREFERRED_DEVICE_TYPE` to compile for the defined `DEVICE_ARCH`
* If `PREFERRED_DEVICE_TYPE` is not specified on build, JIT compilation is assumed and the value of `DEVICE_ARCH` is ignored
* Options for `PREFERRED_DEVICE_TYPE` are `GPU`, `CPU`, or `FPGA`
* The environment variable must be also set before running any code using this lib. The runtime needs this hint to select the right device the code was compile for. If the value was not specified or illegal, the runtime applies a default selection strategy, preferring GPUs over CPUs and CPUs over the host. This might crash the application if the targeted compilation architecture differs from the runtime target
* If `PREFERRED_DEVICE_TYPE` was not specified on build but before running an application, the JIT compiler will generate the kernels and allows switching the device type at runtime
* Complete example call: `export PREFERRED_DEVICE_TYPE=GPU` and `cmake .. -DDEVICE_BACKEND:STRING=oneapi -DREAL_SIZE_IN_BYTES=4 -DDEVICE_ARCH=dg1`

### Device options for AdaptiveCpp
* use `-DDEVICE_BACKEND:STRING=acpp` to build for AdaptiveCpp
* AdaptiveCpp does currently not require the definition of a sub architecture, but it has to be specified in the cmake
*  Complete example call: `cmake .. -DDEVICE_BACKEND:STRING=acpp -DREAL_SIZE_IN_BYTES=4 -DDEVICE_ARCH=dg1`

## Add another API
* Extend the CMakeLists.txt with the new API
* Add a vendor specific sub cmake that is included in the CMakeLists.txt
* Do the same for the examples/basic and examples/jacobi build files
* Add a folder for the new API in interfaces/
* Copy for example an existing implementation into the new folder but implement it regarding the new API
* Compile and run the basic folder to get feedback if the basic concepts are working
* Implement examples/jacobi/src/gpu/kernels for your new API
* compile and run the jacobi benchmark
* Now switch to the algorithms package and repeat the procedure
* You can now compile and run the examples in the tests/ folder

## Add another SYCL compiler
* Add the new compiler as Device backend in the `CMakeLists.txt` in root and example/jacobi such that `sycl.cmake` is included for it
* Adjust `sycl.cmake` in both folders such that the sycl environment is correctly set up for the device and kernel target
* If there is a reduction implementation for this compiler, add it in `algorithms` and extend the if/else in the root `CMakeLists.txt` 
