#include <iostream>
#include "single_gemm_driver.h"

int main() {

// global sizes of matrices
    const unsigned M = 56; 
    const unsigned N = 9;
    const unsigned K = 9;

    // patch sizes of the global matrices 
    const unsigned m = 56; 
    const unsigned n = 9;
    const unsigned k = 9;

    const unsigned num_elements = 100000;
    const unsigned num_repeats = 100; 

    // matrix configurations of patches for GEMM
    // NOTE: HERE IS THE TEST DIFFERENCE BETWEEN single_gemm_xx.cpp files
    CBLAS_LAYOUT Layout = CblasColMajor;
    CBLAS_TRANSPOSE transa = CblasNoTrans;
    CBLAS_TRANSPOSE transb = CblasNoTrans;
    

    SingleGemmCudaDriver Test(Layout, transa, transb,
                              M, m, 
                              N, n, 
                              K, k,
                              num_elements,
                              num_repeats);


    //Test.fillMemoryWithRandomNumbers();

    Test.runReferenceImplTest();
    Test.runNaiveDeviceImplWithoutTransfers();
    Test.runNaiveDeviceImplOnlyTransfers();
    Test.runNaiveDeviceImplWithTransfers();
    Test.runNaiveDeviceImplReducedTransfers(5);
    //Test.runPiplineDeviceImplTest();

    return 0;
}