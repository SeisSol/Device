//DEBUGGING
#include <stdio.h>
#include "common.h"


void * device_malloc(size_t size) {
    void *devPtr;
    cudaMalloc(&devPtr, size); CUDA_CHECK;
    return devPtr;
}

void * device_malloc_pinned(size_t size) {
    void *devPtr;
    cudaMallocHost(&devPtr, size); CUDA_CHECK;
    return devPtr;
}


void device_free(void *devPtr) {
    cudaFree(devPtr); CUDA_CHECK;
}

void device_free_pinned(void *devPtr) {
    cudaFreeHost(devPtr); CUDA_CHECK;
}



void device_copy_to(void* dst, const void* src, size_t count) {
    cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice); CUDA_CHECK;
}


void device_copy_from(void* dst, const void* src, size_t count) {
    cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost); CUDA_CHECK;
}

void device_copy_between(void* dst, const void* src, size_t count) {
    cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice); CUDA_CHECK;
}

void device_copy_from_asynch(void* dst, const void* src, size_t count) {
    //cudaStream_t *stream = new cudaStream_t;
    //cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking); CUDA_CHECK;
    //cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, *stream); CUDA_CHECK;
    //return reinterpret_cast<void*>(stream);

    cudaStream_t stream; 
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); CUDA_CHECK;
    cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream); CUDA_CHECK;
    cudaStreamDestroy(stream); CUDA_CHECK;
}


//void device_asynch_copy_finish(void *handler) {
    //cudaStreamDestroy(*(reinterpret_cast<cudaStream_t*>(handler)));    
//}


void device_copy_2D_to(void *dst,
                       size_t dpitch,
                       const void *src,
                       size_t spitch,
                       size_t width,
                       size_t height) {

    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice); CUDA_CHECK;
}

void device_copy_2D_from(void *dst,
                         size_t dpitch,
                         const void *src,
                         size_t spitch,
                         size_t width,
                         size_t height) {
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost); CUDA_CHECK;
}





void device_compare_with_host_array(const real *host_ptr,
                                    const real * dev_ptr,
                                    const size_t num_elements,
                                    const char *array_name) {

    if (array_name)
        printf("DEVICE: comparing array: %s\n", array_name);

    real * temp = new real[num_elements];
    cudaMemcpy(temp, dev_ptr, num_elements * sizeof(real), cudaMemcpyDeviceToHost); CUDA_CHECK;
    const real eps = 1e-12;
    for (unsigned i = 0; i < num_elements; ++i) {
        if (abs(host_ptr[i] - temp[i]) > eps) {
            printf("ERROR::DEVICE:: host and device arrays are different\n");
            printf("ERROR::DEVICE:: host array (%f) != device array(%f); index (%d)\n", host_ptr[i], temp[i], i);
            delete [] temp;
            abort();
        }
        /*
        else {
            printf("DEVICE: values %f == %f (%s)\n", host_ptr[i], temp[i], array_name);
        }
        */
    }
    printf("DEVICE: host and device arrays are the same\n\n");

    delete [] temp;
}
