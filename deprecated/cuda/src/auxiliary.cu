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
        throw;
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
    cudaDeviceSynchronize(); CUDA_CHECK;
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


// ------------------------------------------------------------------------------
__global__ void kernel_vector_copy_add(unsigned* dst_ptr,
                                       unsigned* src_ptr,
                                       unsigned scalar,
                                       size_t num_elements)  {
  unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < num_elements) {
    dst_ptr[index] = src_ptr[index] + scalar;
  }
}
void device_vector_copy_add(unsigned* dst_ptr, unsigned* src_ptr, unsigned scalar, size_t num_elements) {
  unsigned optimal_kernel_size = 32;
  dim3 block(optimal_kernel_size, 1, 1);
  dim3 grid = compute_grid_1D(block,  num_elements);
  kernel_vector_copy_add<<<grid, block>>>(dst_ptr, src_ptr, scalar, num_elements); CUDA_CHECK;
}



// ------------------------------------------------------------------------------
/**
 * Used to check whether memory was allocated as expected.
 * NOTE: it is going to get crashed in case of inconsistent indices. It means that binning was dont incorrectly
 */
__global__ void kernel_touch_variables(real* base_ptr,
                                       unsigned *indices,
                                       unsigned var_size)  {
  unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
  real *variable = base_ptr + indices[blockIdx.x];
  const real DUMMY_VALUE = 2e-16;
  if (index < var_size) {
    // addition prevents the compiler optimization
    variable[index] += DUMMY_VALUE;
  }
}

/**
 * This piece of code is not for production! Only for debugging the memory allocation
 */
void device_touch_variables(real* base_ptr, unsigned *indices, unsigned var_size, size_t num_vatiables) {
  dim3 block(var_size, 1, 1);
  dim3 grid (num_vatiables, 1, 1); // grid(x, y, z)
  kernel_touch_variables<<<grid, block>>>(base_ptr, indices, var_size); CUDA_CHECK;
}

// ------------------------------------------------------------------------------
__global__ void kernel_stream_data(real **base_src,
                                   real **base_dst,
                                   unsigned element_size) {

    real *src_element = base_src[blockIdx.x];
    real *dst_element = base_dst[blockIdx.x];

    if (threadIdx.x < element_size) {
      dst_element[threadIdx.x] = src_element[threadIdx.x];
    }
}


void device_stream_data(real **base_src,
                        real **base_dst,
                        unsigned element_size,
                        unsigned num_elements) {

  dim3 block(element_size, 1, 1);
  dim3 grid (num_elements, 1, 1); // grid(x, y, z)
  kernel_stream_data<<<grid, block>>>(base_src, base_dst, element_size); CUDA_CHECK;
}



// ------------------------------------------------------------------------------
__global__ void kernel_accumulate_data(real **base_src,
                                       real **base_dst,
                                       unsigned element_size) {

  real *src_element = base_src[blockIdx.x];
  real *dst_element = base_dst[blockIdx.x];

  if (threadIdx.x < element_size) {
    dst_element[threadIdx.x] += src_element[threadIdx.x];
  }
}

void device_accumulate_data(real **base_src,
                            real **base_dst,
                            unsigned element_size,
                            unsigned num_elements) {

  dim3 block(element_size, 1, 1);
  dim3 grid (num_elements, 1, 1); // grid(x, y, z)
  kernel_accumulate_data<<<grid, block>>>(base_src, base_dst, element_size); CUDA_CHECK;
}
