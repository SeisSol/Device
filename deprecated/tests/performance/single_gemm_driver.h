#ifndef SINGLE_GEMM_CUDA_DRIVER_H_
#define SINGLE_GEMM_CUDA_DRIVER_H_

#include <mkl.h>
#include "helper.h"

class SingleGemmCudaDriver {
public:
    SingleGemmCudaDriver(const CBLAS_LAYOUT u_Layout,
                         const CBLAS_TRANSPOSE u_transa,
                         const CBLAS_TRANSPOSE u_transb,
                         const unsigned u_M, const unsigned u_m,
                         const unsigned u_N, const unsigned u_n,
                         const unsigned u_K, const unsigned u_k,
                         const unsigned u_num_elements,
                         const unsigned u_num_repeats);

    ~SingleGemmCudaDriver();

    void prepareTest();
    void set_str_addressA(const unsigned rowIdx, const unsigned columnIdx);
    void set_str_addressB(const unsigned rowIdx, const unsigned columnIdx);
    void set_str_addressC(const unsigned rowIdx, const unsigned columnIdx);

    void fillMemoryWithRandomNumbers();
    void fillMemoryWith(real c_value, real a_value, real b_value);

    void changeSubPatchSizes(const unsigned u_m, const unsigned u_n, const unsigned u_k);
    void runReferenceImplTest();
    void runNaiveDeviceImplWithoutTransfers();
    void runNaiveDeviceImplOnlyTransfers();
    void runNaiveDeviceImplWithTransfers();
    void runNaiveDeviceImplReducedTransfers(const unsigned ratio);


    void runPiplineDeviceImplTest();

private:

    void checkTestParameters();

    CBLAS_LAYOUT Layout;
    CBLAS_TRANSPOSE transa;
    CBLAS_TRANSPOSE transb;

    unsigned M, N, K;
    unsigned size_C;
    unsigned size_A;
    unsigned size_B;  

    unsigned m, n, k;
    unsigned start_ptr_A;
    unsigned start_ptr_B;
    unsigned start_ptr_C;
    unsigned lda, ldb, ldc;

    unsigned num_elements;
    unsigned num_repeats;

    real alpha, beta;

    real* C;
    real* C_cuda_result;
    real* A;
    real* B;

    real* d_C;
    real* d_B;
    real* d_A;
};


#endif  // SINGLE_GEMM_CUDA_DRIVER_H_