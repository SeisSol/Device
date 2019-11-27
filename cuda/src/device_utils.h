#ifndef CUDA_UTILS_CUH_
#define CUDA_UTILS_CUH_

#include <string>
#include "common.h"
#include "temporary_mem_menager.h"
#include "nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
        int color_id = cid; \
        color_id = color_id%num_colors;\
        nvtxEventAttributes_t eventAttrib = {0}; \
        eventAttrib.version = NVTX_VERSION; \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        eventAttrib.colorType = NVTX_COLOR_ARGB; \
        eventAttrib.color = colors[color_id]; \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name; \
        nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();



// memory menagment:
void check_device_operation();
void * device_malloc(size_t size);
void * device_malloc_unified(size_t size);
void * device_malloc_pinned(size_t size);

void device_free(void *devPtr);
void device_free_pinned(void *devPtr);
void device_synch();


void device_copy_to(void* dst, const void* src, size_t count);
void device_copy_from(void* dst, const void* src, size_t count);
void device_copy_between(void* dst, const void* src, size_t count);


void device_copy_from_asynch(void* dst, const void* src, size_t count); 
//void device_asynch_copy_finish(void *handler); 


void device_copy_2D_to(void *dst,
                       size_t dpitch,
                       const void *src,
                       size_t spitch,
                       size_t width,
                       size_t height);

void device_copy_2D_from(void *dst,
                         size_t dpitch,
                         const void *src,
                         size_t spitch,
                         size_t width,
                         size_t height);


void device_compare_with_host_array(const real *host_ptr,
                                    const real * dev_ptr,
                                    const size_t num_elements,
                                    const char *array_name = NULL);

void device_init_linspace(const size_t num_elements, unsigned *array);
void device_scale_linspace(const unsigned* linspace, const unsigned scale, const size_t num_elements, unsigned *output);
void device_scale_array(const real scalar, const size_t num_elements, real *dev_array);
void device_scale_array(const unsigned scale, const size_t num_elements, unsigned *output);
void device_init_array(const unsigned value, const size_t num_elements, unsigned *output);
void device_vector_copy_add(unsigned* dst_ptr, unsigned* src_ptr, unsigned scalar, size_t num_elements);

void device_stream_data(real *base_src, unsigned *src_indices,
                        real *base_dst, unsigned *dst_indices,
                        unsigned element_size, unsigned num_elements);

void device_accumulate_data(real *base_src, unsigned *src_indices,
                            real *base_dst, unsigned *dst_indices,
                            unsigned element_size, unsigned num_elements);


void device_touch_variables(real* base_ptr, unsigned *indices, unsigned var_size, size_t num_vatiables);

// computational kernels:
#if !defined CBLAS_H && !defined __MKL_CBLAS_H__
    enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102};
    enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
#endif


template <typename T, typename D>
void device_copy_add_scale(const int m, const int n,
                           const real alpha, const real *A, const int lda,
                           const real beta, real *B, const int ldb,
                           T stride_A,
                           D stride_B,
                           const unsigned num_elements);


template <typename T, typename D, typename F>
void device_gemm(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE transa,
                 const CBLAS_TRANSPOSE transb,
                 const int m, const int n, const int k,
                 const real alpha, const real *A_base, const int lda,
                 const real *B_base, const int ldb,
                 const real beta, real *C_base, const int ldc,
                 T stride_A,
                 D stride_B,
                 F stride_C,
                 const unsigned num_elements = 1);

#endif  // CUDA_UTILS_CUH_