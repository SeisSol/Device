#ifndef MULTIPLE_GEMM_CUDA_DRIVER_H_
#define MULTIPLE_GEMM_CUDA_DRIVER_H_

#include <mkl.h>
#include "helper.h"

class MultipleGemmCudaDriver {
public:
    MultipleGemmCudaDriver(const CBLAS_LAYOUT u_Layout,
                           const CBLAS_TRANSPOSE u_transa,
                           const CBLAS_TRANSPOSE u_transb,
                           const unsigned u_tensor_A_size[3],
                           const unsigned u_tensor_B_size[3],
                           const unsigned u_tensor_C_size[3],
                           const unsigned u_m,
                           const unsigned u_n,
                           const unsigned u_k,
                           const unsigned u_num_elements);

    ~MultipleGemmCudaDriver();

    void prepareTest();
    void set_str_addressA(const unsigned ld_idx, const unsigned ls_idx, const unsigned lt_idx);
    void set_str_addressB(const unsigned ld_idx, const unsigned ls_idx, const unsigned lt_idx);
    void set_str_addressC(const unsigned ld_idx, const unsigned ls_idx, const unsigned lt_idx);
    void set_leading_dimensions(const unsigned lda, const unsigned ldb, const unsigned ldc);

    void fillMemoryWithRandomNumbers();
    void fillMemoryWith(real c_value, real a_value, real b_value);

    void changeSubPatchSizes(const unsigned u_m, const unsigned u_n, const unsigned u_k);
    bool runTest();

private:

    void checkTestParameters();

    CBLAS_LAYOUT Layout;
    CBLAS_TRANSPOSE transa;
    CBLAS_TRANSPOSE transb;

    unsigned tensor_A_size[3];
    unsigned tensor_B_size[3];
    unsigned tensor_C_size[3];

    unsigned volume_tensor_A;
    unsigned volume_tensor_B;
    unsigned volume_tensor_C;

    unsigned num_elements;

    unsigned m, n, k;
    unsigned start_ptr_A;
    unsigned start_ptr_B;
    unsigned start_ptr_C;
    unsigned lda, ldb, ldc;

    real alpha, beta;

    real* C;
    real* C_cuda_result;
    real* A;
    real* B;

    real* d_C;
    real* d_B;
    real* d_A;
};


#endif  // GEMM_CUDA_TESTCASE_H_