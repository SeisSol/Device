#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <assert.h>


#include "single_gemm_driver.h"
#include <device_utils.h>

SingleGemmCudaDriver::SingleGemmCudaDriver(const CBLAS_LAYOUT u_Layout,
                                           const CBLAS_TRANSPOSE u_transa,
                                           const CBLAS_TRANSPOSE u_transb,
                                           const unsigned u_M, const unsigned u_m,
                                           const unsigned u_N, const unsigned u_n,
                                           const unsigned u_K, const unsigned u_k) : Layout(u_Layout), 
                                                                                     transb(u_transb),
                                                                                     transa(u_transa), 
                                                                                     M(u_M), m(u_m), N(u_N), 
                                                                                     n(u_n), K(u_K), k(u_k),
                                                                                     start_ptr_A(0), lda(u_M),
                                                                                     start_ptr_B(0), ldb(u_K),
                                                                                     start_ptr_C(0), ldc(u_M),
                                                                                     C(NULL), A(NULL), B(NULL),
                                                                                     d_A(NULL), d_B(NULL), d_C(NULL),
                                                                                     C_cuda_result(NULL),
                                                                                     alpha(1.0), beta(1.0) {
    this->prepareTest();
}


SingleGemmCudaDriver::~SingleGemmCudaDriver() {
    if (C) delete [] C;
    if (A) delete [] A;
    if (B) delete [] B;
    if (C_cuda_result) delete [] C_cuda_result;

    if (d_C) {device_free(d_C);}
    if (d_A) {device_free(d_A);}
    if (d_B) {device_free(d_B);}
}


void SingleGemmCudaDriver::changeSubPatchSizes(const unsigned u_m, const unsigned u_n, const unsigned u_k) {
    m = u_m;
    n = u_n;
    k = u_k;
    this->checkTestParameters();
}


void SingleGemmCudaDriver::set_str_addressA(const unsigned rowIdx, 
                                            const unsigned columnIdx) {

    if (transa == CblasNoTrans) {
        assert(m < (M - rowIdx));  // check column gap
        assert(k < (K - columnIdx));  // check column gap
        start_ptr_A = rowIdx + columnIdx * M;
    }
    else if (transa == CblasTrans) {
        assert(k < (M - rowIdx));  // check column gap
        assert(m < (K - columnIdx));  // check column gap
        start_ptr_A = rowIdx + columnIdx * M;
    }
    else {
        std::cerr << "ERROR: unknown value for transa" << std::endl;
    }
} 


void SingleGemmCudaDriver::set_str_addressB(const unsigned rowIdx, 
                                            const unsigned columnIdx) {

    if (transb == CblasNoTrans) {
        assert(k < (K - rowIdx));  // check column gap
        assert(n < (N - columnIdx));  // check column gap
        start_ptr_B = rowIdx + columnIdx * K;
    }
    else if (transb == CblasTrans) {
        assert(n < (K - rowIdx));  // check column gap
        assert(k < (N - columnIdx));  // check column gap
        start_ptr_B = rowIdx + columnIdx * M;
    }
    else {
        std::cerr << "ERROR: unknown value for transa" << std::endl;
    }
}


void SingleGemmCudaDriver::set_str_addressC(const unsigned rowIdx, 
                                            const unsigned columnIdx) {

    assert(m < (M - rowIdx));  // check column gap
    assert(n < (N - columnIdx));  // check column gap
    start_ptr_C = rowIdx + columnIdx * M;
}


void SingleGemmCudaDriver::checkTestParameters() {
    assert(M >= m);
    assert(N >= n);
    assert(K >= k);
}


void SingleGemmCudaDriver::prepareTest() {

    this->checkTestParameters();

    // compute full matrix sizes
    size_C = M * N;
    size_A = M * K;
    size_B = K * N;  

    // compute full matrix sizes
    C = new real[size_C];
    A = new real[size_A];
    B = new real[size_B];
    C_cuda_result = new real[size_C];

    // check whether OS allocated memory for arrays
    assert(C != NULL);
    assert(A != NULL);
    assert(B != NULL);
    assert(C_cuda_result != NULL);

    // allocated data on GPU
    d_C = (real*)device_malloc(size_C * sizeof(real));
    d_A = (real*)device_malloc(size_A * sizeof(real));
    d_B = (real*)device_malloc(size_B * sizeof(real));

    // fill matrices with random numbers
    fillWithStuff(C, size_C, 0.0);
    fillWithStuff(A, size_A, 2.0);
    fillWithStuff(B, size_B, 1.0);

    // move data from CPU to GPU
    device_copy_to(d_C, C, size_C * sizeof(real));
    device_copy_to(d_A, A, size_A * sizeof(real));
    device_copy_to(d_B, B, size_B * sizeof(real));
}


void SingleGemmCudaDriver::fillMemoryWithRandomNumbers() {
    fillWithStuff(C, size_C);
    fillWithStuff(A, size_A);
    fillWithStuff(B, size_B);

    // move data from CPU to GPU
    device_copy_to(d_C, C, size_C * sizeof(real));
    device_copy_to(d_A, A, size_A * sizeof(real));
    device_copy_to(d_B, B, size_B * sizeof(real));
 }


void SingleGemmCudaDriver::fillMemoryWith(real c_value, real a_value, real b_value) {
    fillWithStuff(C, size_C, c_value);
    fillWithStuff(A, size_A, a_value);
    fillWithStuff(B, size_B, b_value);

    // move data from CPU to GPU
    device_copy_to(d_C, C, size_C * sizeof(real));
    device_copy_to(d_A, A, size_A * sizeof(real));
    device_copy_to(d_B, B, size_B * sizeof(real));
}


bool SingleGemmCudaDriver::runTest() {

    // call intel mkl
    MKL_GEMM(Layout, transa, transb, 
             m, n, k, 
             alpha, (A + start_ptr_A) , lda, 
             (B + start_ptr_B), ldb, 
             beta, (C + start_ptr_C), ldc);

    // call cuda
    const unsigned strideA = 0, strideB = 0, strideC = 0;
    cuda_blas_gemm(Layout, transa, transb, 
                   m, n, k, 
                   alpha, d_A + start_ptr_A, lda, 
                   d_B + start_ptr_B, ldb,
                   beta, d_C + start_ptr_C, ldc,
                   strideA,
                   strideB,
                   strideC,
                   1);
    



    // move data back from GPU to CPU
    device_copy_from(C_cuda_result, d_C, size_C * sizeof(real));

    // compare results
    std::cout << std::fixed << std::setprecision(12);
    bool is_same_result = true;
    for (int i = 0; i < size_C; ++i) {
#ifdef FAST_COMPARISON         
        if (std::abs(C[i] - C_cuda_result[i]) > EPS) {
            std::cout.precision(5);
            std::cout << "ERROR: linear elem: " << i << ";"
                      << " mkl: " << C[i] << " cuda " << C_cuda_result[i] << std::endl;
            is_same_result = false;
        }
 #else
        const unsigned incorrect_bits = count_incorrect_bits(C[i], C_cuda_result[i]);
        if (incorrect_bits > TOLERANCE) {

            real abs_diff = (std::abs(C[i] - C_cuda_result[i]));
            real rel_diff = 100 * abs_diff / std::abs(C[i]);

            std::cout << "ERROR: linear elem: " << i << ";" 
                      << " num. inc. bits: " << incorrect_bits << ";"
                      << " rel. diff, %: " << rel_diff
                      << " abs. diff, " << abs_diff
                      << std::endl;

            std::cout << " mkl: " << C[i] 
                      << " device " << C_cuda_result[i]
                      << std::endl; 

                
            is_same_result = false;
        }
#endif                
    }

    return is_same_result;
}