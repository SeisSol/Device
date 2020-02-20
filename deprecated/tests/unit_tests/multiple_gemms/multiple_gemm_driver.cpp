#include <mkl.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <assert.h>

#include "multiple_gemm_driver.h"
#include "device_utils.h"

MultipleGemmCudaDriver::MultipleGemmCudaDriver(const CBLAS_LAYOUT u_Layout,
                                               const CBLAS_TRANSPOSE u_transa,
                                               const CBLAS_TRANSPOSE u_transb,
                                               const unsigned u_tensor_A_size[3],
                                               const unsigned u_tensor_B_size[3],
                                               const unsigned u_tensor_C_size[3],
                                               const unsigned u_m,
                                               const unsigned u_n,
                                               const unsigned u_k,
                                               const unsigned u_num_elements) 
                                               : Layout(u_Layout), 
                                                 transb(u_transb),
                                                 transa(u_transa),
                                                 num_elements(u_num_elements),
                                                 m(u_m), n(u_n), k(u_k),
                                                 start_ptr_A(0), lda(u_tensor_A_size[0]),
                                                 start_ptr_B(0), ldb(u_tensor_B_size[0]),
                                                 start_ptr_C(0), ldc(u_tensor_C_size[0]),
                                                 C(NULL), A(NULL), B(NULL),
                                                 d_A(NULL), d_B(NULL), d_C(NULL),
                                                 C_cuda_result(NULL),
                                                 alpha(1.0), beta(1.0) {

    volume_tensor_A = 1;
    volume_tensor_B = 1;
    volume_tensor_C = 1;

    for (unsigned i = 0; i < 3; ++i) {
        assert(u_tensor_A_size[i] != 0);
        assert(u_tensor_B_size[i] != 0);
        assert(u_tensor_C_size[i] != 0);

        tensor_A_size[i] = u_tensor_A_size[i];
        tensor_B_size[i] = u_tensor_B_size[i];
        tensor_C_size[i] = u_tensor_C_size[i];

        volume_tensor_A *= tensor_A_size[i];
        volume_tensor_B *= tensor_B_size[i];
        volume_tensor_C *= tensor_C_size[i]; 
    }

    assert(num_elements != 0);

    this->prepareTest();
}

MultipleGemmCudaDriver::~MultipleGemmCudaDriver() {
    if (C) delete [] C;
    if (A) delete [] A;
    if (B) delete [] B;
    if (C_cuda_result) delete [] C_cuda_result;

    if (d_C) {device_free(d_C);}
    if (d_A) {device_free(d_A);}
    if (d_B) {device_free(d_B);}
}


void MultipleGemmCudaDriver::changeSubPatchSizes(const unsigned u_m, 
                                                 const unsigned u_n, 
                                                 const unsigned u_k) {
    m = u_m;
    n = u_n;
    k = u_k;
    this->checkTestParameters();
}


void MultipleGemmCudaDriver::set_str_addressA(const unsigned ld_idx, 
                                              const unsigned ls_idx, 
                                              const unsigned lt_idx) {
    start_ptr_A = ld_idx + ls_idx * tensor_A_size[0] + lt_idx * tensor_A_size[1] * tensor_A_size[2];
} 


void MultipleGemmCudaDriver::set_str_addressB(const unsigned ld_idx, const unsigned ls_idx, const unsigned lt_idx) {
    start_ptr_B = ld_idx + ls_idx * tensor_B_size[0] + lt_idx * tensor_B_size[1] * tensor_B_size[2];
}



void MultipleGemmCudaDriver::set_str_addressC(const unsigned ld_idx, const unsigned ls_idx, const unsigned lt_idx) {
    start_ptr_C = ld_idx + ls_idx * tensor_C_size[0] + lt_idx * tensor_C_size[1] * tensor_C_size[2];
}

void MultipleGemmCudaDriver::set_leading_dimensions(const unsigned u_lda,
                                                    const unsigned u_ldb, 
                                                    const unsigned u_ldc) {
    lda = u_lda;
    ldb = u_ldc;
    ldc = u_ldc;

    this->checkTestParameters();
}


void MultipleGemmCudaDriver::checkTestParameters() {
    const int num_rows_A = CblasNoTrans ? m : k;
    const int num_rows_B = CblasNoTrans ? k : n;

    assert((volume_tensor_A >= lda) && (lda >= num_rows_A));
    assert((volume_tensor_B >= ldb) && (lda >= num_rows_B));
    assert((volume_tensor_C >= ldc) && (ldc >= m));
}


void MultipleGemmCudaDriver::prepareTest() {

    this->checkTestParameters();

    // compute full matrix sizes
    A = new real[num_elements * volume_tensor_A];
    B = new real[num_elements * volume_tensor_B];
    C = new real[num_elements * volume_tensor_C];
    C_cuda_result = new real[num_elements * volume_tensor_C];

    // check whether OS allocated memory for arrays
    assert(C != NULL);
    assert(A != NULL);
    assert(B != NULL);
    assert(C_cuda_result != NULL);

    // allocated data on a device
    d_A = (real*)device_malloc(num_elements * volume_tensor_A * sizeof(real));
    d_B = (real*)device_malloc(num_elements * volume_tensor_B * sizeof(real));
    d_C = (real*)device_malloc(num_elements * volume_tensor_C * sizeof(real));


    // fill matrices with random numbers
    fillWithStuff(A, num_elements * volume_tensor_A, 2.0);
    fillWithStuff(B, num_elements * volume_tensor_B, 1.0);
    fillWithStuff(C, num_elements * volume_tensor_C, 0.0);

    // move data from host to device
    device_copy_to(d_A, A, num_elements * volume_tensor_A * sizeof(real));
    device_copy_to(d_B, B, num_elements * volume_tensor_B * sizeof(real));
    device_copy_to(d_C, C, num_elements * volume_tensor_C * sizeof(real));
}


void MultipleGemmCudaDriver::fillMemoryWithRandomNumbers() {
    fillWithStuff(A, num_elements * volume_tensor_A);
    fillWithStuff(B, num_elements * volume_tensor_B);
    fillWithStuff(C, num_elements * volume_tensor_C);

    // move data from host to device
    device_copy_to(d_A, A, num_elements * volume_tensor_A * sizeof(real));
    device_copy_to(d_B, B, num_elements * volume_tensor_B * sizeof(real));
    device_copy_to(d_C, C, num_elements * volume_tensor_C * sizeof(real));
}


void MultipleGemmCudaDriver::fillMemoryWith(real c_value, real a_value, real b_value) {
    fillWithStuff(A, num_elements * volume_tensor_A, a_value);
    fillWithStuff(B, num_elements * volume_tensor_B, b_value);
    fillWithStuff(C, num_elements * volume_tensor_C, c_value);

    // move data from host to device
    device_copy_to(d_A, A, num_elements * volume_tensor_A * sizeof(real));
    device_copy_to(d_B, B, num_elements * volume_tensor_B * sizeof(real));
    device_copy_to(d_C, C, num_elements * volume_tensor_C * sizeof(real));
}


bool MultipleGemmCudaDriver::runTest() {

    // call intel mkl
    
    for (unsigned element = 0; element < num_elements; ++ element) {
        real *next_A = A + start_ptr_A + element * volume_tensor_A;
        real *next_B = B + start_ptr_B + element * volume_tensor_B;
        real *next_C = C + start_ptr_C + element * volume_tensor_C;

        /*
        cblas_dgemm(Layout, transa, transb, 
                    m, n, k, 
                    alpha, next_A, lda, 
                    next_B, ldb, 
                    beta, next_C, ldc);
        */
        MKL_GEMM(Layout, transa, transb, 
                 m, n, k, 
                 alpha, next_A, lda, 
                 next_B, ldb, 
                 beta, next_C, ldc);
    }

    
    // call cuda
    cuda_blas_gemm(Layout, transa, transb, 
                   m, n, k, 
                   alpha, d_A + start_ptr_A, lda, 
                   d_B + start_ptr_B, ldb,
                   beta, d_C + start_ptr_C, ldc,
                   volume_tensor_A,
                   volume_tensor_B,
                   volume_tensor_C,
                   num_elements);
    


    // move data back from GPU to CPU
    device_copy_from(C_cuda_result, d_C, num_elements * volume_tensor_C * sizeof(real));


    // compare results
    std::cout << std::fixed << std::setprecision(12);
    bool is_same_result = true;
    
    for (unsigned element = 0; element < num_elements; ++element) {
        for (unsigned i = 0; i < volume_tensor_C; ++i) {
            const unsigned index = i + element * volume_tensor_C;
#ifdef FAST_COMPARISON 
            if (std::abs(C[index] - C_cuda_result[index]) > EPS) {
                std::cout.precision(5);
                std::cout << "ERROR: linear elem: " << element << " index: " << i << ";"
                        << " mkl: " << C[index] << " cuda " << C_cuda_result[index] << std::endl;
                is_same_result = false;
            }
#else            
            const unsigned incorrect_bits = count_incorrect_bits(C[index], C_cuda_result[index]);
            if (incorrect_bits > TOLERANCE) {

                real abs_diff = (std::abs(C[index] - C_cuda_result[index]));
                real rel_diff = 100 * abs_diff / std::abs(C[index]);

                std::cout << "ERROR: linear elem: " << element 
                          << " index: " << i << ";" 
                          << " num. inc. bits: " << incorrect_bits << ";"
                          << " rel. diff, %: " << rel_diff
                          << " abs. diff, " << abs_diff
                          << std::endl;

                std::cout << " mkl: " << C[index] 
                          << " device " << C_cuda_result[index]
                          << std::endl; 

                
                is_same_result = false;
            }
#endif 
        }
    }


    return is_same_result;
}