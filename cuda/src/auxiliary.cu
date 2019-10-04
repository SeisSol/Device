#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <stdio.h>


#include "common.h"

#ifndef CUDA_CHECK
    #define CUDA_CHECK cuda_check(__FILE__,__LINE__)
#endif


std::string prev_file = "";
int prev_line = 0;
void cuda_check(std::string file, int line) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << std::endl << file << ", line " << line 
                  << ": " << cudaGetErrorString(e) << " (" << e << ")" << std::endl;
        if (prev_line > 0)
            std::cout << "Previous CUDA call:" << std::endl 
                      << prev_file << ", line " << prev_line << std::endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}


inline dim3 compute_grid_1D(const dim3 &block, const int size) {
    int num_blocks_x = (size + block.x - 1) / block.x;
    return dim3(num_blocks_x, 1, 1);
}


// ------------------------------------------------------------------------------
__global__ void kernel_gpu_operation_check() {
    printf("gpu offloading is working\n");
}

void check_device_operation() {
    std::cout << "checking gpu offloading" << std::endl;
    kernel_gpu_operation_check<<<1,1>>>(); CUDA_CHECK;
    cudaDeviceSynchronize();
}
// ------------------------------------------------------------------------------
void device_synch() {
    cudaDeviceSynchronize();
}

// ------------------------------------------------------------------------------
__global__ void kernel_scale_array(const real scalar, real *array, const size_t num_elements) {
    unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_elements) {
        array[index] *= scalar;
    }
}


void device_scale_array(const real scalar, const size_t num_elements, real *dev_array) {
    unsigned optimal_kernel_size = 32;
    dim3 block(optimal_kernel_size, 1, 1);
    dim3 grid = compute_grid_1D(block,  num_elements);
    kernel_scale_array<<<grid, block>>>(scalar, dev_array, num_elements); CUDA_CHECK;
}

// ------------------------------------------------------------------------------
__global__ void kernel_init_linspace(const size_t num_elements, unsigned *array) {
  unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < num_elements) {
    array[index] = index;
  }
}

void device_init_linspace(const size_t num_elements, unsigned *array) {
  unsigned optimal_kernel_size = 32;
  dim3 block(optimal_kernel_size, 1, 1);
  dim3 grid = compute_grid_1D(block,  num_elements);
  kernel_init_linspace<<<grid, block>>>(num_elements, array); CUDA_CHECK;
}


// ------------------------------------------------------------------------------
__global__ void kernel_scale_linspace(const unsigned* linspace,
                                      const unsigned scale,
                                      const size_t num_elements,
                                      unsigned *output) {
  unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < num_elements) {
    output[index] = scale * linspace[index];
  }
}


void device_scale_linspace(const unsigned* linspace, const unsigned scale, const size_t num_elements, unsigned *output) {
  unsigned optimal_kernel_size = 32;
  dim3 block(optimal_kernel_size, 1, 1);
  dim3 grid = compute_grid_1D(block,  num_elements);
  kernel_scale_linspace<<<grid, block>>>(linspace, scale, num_elements, output); CUDA_CHECK;
}



// ------------------------------------------------------------------------------
__global__ void kernel_scale_array(const unsigned scale,
                                   const size_t num_elements,
                                   unsigned *output)  {
  unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < num_elements) {
    output[index] = scale * output[index];
  }
}

void device_scale_array(const unsigned scale, const size_t num_elements, unsigned *output) {
  unsigned optimal_kernel_size = 32;
  dim3 block(optimal_kernel_size, 1, 1);
  dim3 grid = compute_grid_1D(block,  num_elements);
  kernel_scale_array<<<grid, block>>>(scale, num_elements, output); CUDA_CHECK;
}


// ------------------------------------------------------------------------------
__global__ void kernel_init_array(const unsigned value,
                                  const size_t num_elements,
                                  unsigned *output)  {
  unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < num_elements) {
    output[index] = value;
  }
}
void device_init_array(const unsigned value, const size_t num_elements, unsigned *output) {
  unsigned optimal_kernel_size = 32;
  dim3 block(optimal_kernel_size, 1, 1);
  dim3 grid = compute_grid_1D(block,  num_elements);
  kernel_init_array<<<grid, block>>>(value, num_elements, output); CUDA_CHECK;
}
