#include "device_utils.h"



void * device_malloc(size_t size) {
    void *devPtr;
    cudaMalloc(&devPtr, size); CUDA_CHECK;
    return devPtr;
}


void device_copy_to(void* dst, const void* src, size_t count) {
    cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice); CUDA_CHECK;
}


void device_copy_from(void* dst, const void* src, size_t count) {
    cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost); CUDA_CHECK;
}


void device_free(void *devPtr) {
    cudaFree(devPtr); CUDA_CHECK;
}