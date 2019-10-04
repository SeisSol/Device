#ifndef MULTIPLE_GEMM_CUDA_DRIVER_H_
#define MULTIPLE_GEMM_CUDA_DRIVER_H_

#include "helper.h"

class CopyAddScaleDriver {
public:
    CopyAddScaleDriver(const unsigned u_tensor_A_size[2],
                       const unsigned u_tensor_B_size[2],
                       const unsigned u_m,
                       const unsigned u_n,
                       const unsigned u_num_elements);
    
    ~CopyAddScaleDriver();
    
    void prepareTest();
    void fillMemoryWithRandomNumbers();
    void fillMemoryWith(real c_value, real a_value, real b_value);
    void set_scales(const float u_alpha, float const u_beta);

    //old changeSubPatchSizes(const unsigned u_m, const unsigned u_n, const unsigned u_k);
    bool runTest();
    
private:

    void checkTestParameters();   

    unsigned tensor_A_size[2];
    unsigned tensor_B_size[2];
    
    unsigned volume_tensor_A;
    unsigned volume_tensor_B;

    unsigned num_elements;

    unsigned m, n;
    unsigned lda, ldb;

    real alpha, beta;

    real* B;
    real* B_cuda_result;
    real* A;

    real* d_B;
    real* d_A;
    
};


#endif  // GEMM_CUDA_TESTCASE_H_