#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include "copy_add_scale_driver.h"
#include "device_utils.h"

CopyAddScaleDriver::CopyAddScaleDriver(const unsigned u_tensor_A_size[2],
                                       const unsigned u_tensor_B_size[2],
                                       const unsigned u_m,
                                       const unsigned u_n,
                                       const unsigned u_num_elements) : m(u_m), 
                                                                        n(u_n),
                                                                        lda(u_tensor_A_size[0]),
                                                                        ldb(u_tensor_B_size[0]),
                                                                        alpha(1.0),
                                                                        beta(1.0),
                                                                        num_elements(u_num_elements)
{

    volume_tensor_A = 1;
    volume_tensor_B = 1;

    for (unsigned i = 0; i < 2; ++i) {
        assert(u_tensor_A_size[i] != 0);
        assert(u_tensor_B_size[i] != 0);

        tensor_A_size[i] = u_tensor_A_size[i];
        tensor_B_size[i] = u_tensor_B_size[i];

        volume_tensor_A *= tensor_A_size[i];
        volume_tensor_B *= tensor_B_size[i];
    }

    this->prepareTest();
}


CopyAddScaleDriver::~CopyAddScaleDriver() {
    if (A) delete [] A;
    if (B) delete [] B;
    if (B_cuda_result) delete [] B_cuda_result;

    if (d_A) {device_free(d_A);}
    if (d_B) {device_free(d_B);}
}


void CopyAddScaleDriver::checkTestParameters() {
    assert(lda >= m);
    assert(ldb >= n);
}


void CopyAddScaleDriver::prepareTest() {

    this->checkTestParameters();

    // compute full matrix sizes
    A = new real[num_elements * volume_tensor_A];
    B = new real[num_elements * volume_tensor_B];
    B_cuda_result = new real[num_elements * volume_tensor_B];

    // check whether OS allocated memory for arrays
    assert(A != NULL);
    assert(B != NULL);
    assert(B_cuda_result != NULL);

    // allocated data on GPU

    d_A = (real*)device_malloc(num_elements * volume_tensor_A * sizeof(real));
    d_B = (real*)device_malloc(num_elements * volume_tensor_B * sizeof(real));
    
    // fill matrices with random numbers
    fillWithStuff(A, num_elements * volume_tensor_A, 2.0);
    fillWithStuff(B, num_elements * volume_tensor_B, 1.0);
    
    // move data from CPU to GPU

    device_copy_to(d_A, A, num_elements * volume_tensor_A * sizeof(real));
    device_copy_to(d_B, B, num_elements * volume_tensor_B * sizeof(real));
}


void CopyAddScaleDriver::fillMemoryWithRandomNumbers() {
    fillWithStuff(A, num_elements * volume_tensor_A);
    fillWithStuff(B, num_elements * volume_tensor_B);

    // move data from CPU to GPU
    device_copy_to(d_A, A, num_elements * volume_tensor_A * sizeof(real));
    device_copy_to(d_B, B, num_elements * volume_tensor_B * sizeof(real));
}


void CopyAddScaleDriver::fillMemoryWith(real c_value, real a_value, real b_value) {
    fillWithStuff(A, num_elements * volume_tensor_A, a_value);
    fillWithStuff(B, num_elements * volume_tensor_B, b_value);

    // move data from CPU to GPU
    device_copy_to(d_A, A, num_elements * volume_tensor_A * sizeof(real));
    device_copy_to(d_B, B, num_elements * volume_tensor_B * sizeof(real));
}


void CopyAddScaleDriver::set_scales(const float u_alpha, float const u_beta) {
    alpha = u_alpha;
    beta = u_beta;
}


bool CopyAddScaleDriver::runTest() {

 
    // run naive implementation   
    for (unsigned element = 0; element < num_elements; ++ element) {
        real *next_A = A + element * volume_tensor_A;
        real *next_B = B + element * volume_tensor_B;

        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                next_B[i + j * ldb] = beta * next_B[i + j * ldb] + alpha * next_A[i + j * lda];
            }
        }
    }

    
    // call cuda
    cuda_copy_add_scale(m, n, 
                        alpha, d_A, lda, 
                        beta, d_B, ldb,
                        volume_tensor_A,
                        volume_tensor_B,
                        num_elements);

    
    
    // move data back from GPU to CPU
    device_copy_from(B_cuda_result, d_B, num_elements * volume_tensor_B * sizeof(real));


    // compare results
    std::cout << std::fixed << std::setprecision(12);
    bool is_same_result = true;    
    for (unsigned element = 0; element < num_elements; ++element) {
        for (unsigned i = 0; i < volume_tensor_B; ++i) {

            const unsigned index = i + element * volume_tensor_B;
#ifdef FAST_COMPARISON                        
            if (std::abs(B[index] - B_cuda_result[index]) > EPS) {
                std::cout.precision(5);
                std::cout << "ERROR: linear elem: " << element << " index: " << i << ";"
                        << " mkl: " << B[index] << " cuda " << B_cuda_result[index] << std::endl;
                is_same_result = false;
            }
#else
            const unsigned incorrect_bits = count_incorrect_bits(B[index], B_cuda_result[index]);
            if (incorrect_bits > TOLERANCE) {

            real abs_diff = (std::abs(B[index] - B_cuda_result[index]));
            real rel_diff = 100 * abs_diff / std::abs(B[index]);

            std::cout << "ERROR: linear elem: " << element 
                      << " index: " << i << ";" 
                      << " num. inc. bits: " << incorrect_bits << ";"
                      << " rel. diff, %: " << rel_diff
                      << " abs. diff, " << abs_diff
                      << std::endl;

            std::cout << " mkl: " << B[index] 
                      << " device " << B_cuda_result[index]
                      << std::endl; 
                
            is_same_result = false;
            }
#endif
        }
    }


    return is_same_result;
}