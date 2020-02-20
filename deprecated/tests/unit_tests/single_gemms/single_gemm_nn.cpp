#include <catch2/catch.hpp>

#include "single_gemm_driver.h"


TEST_CASE( "Gemm on CPU and GPU - A=CblasNoTrans, B=CblasNoTrans", "[GemmTest_NN]" ) {
    // global sizes of matrices
    const unsigned M = 61; 
    const unsigned N = 52;
    const unsigned K = 73;

    // patch sizes of the global matrices 
    unsigned m = 8; 
    unsigned n = 5;
    unsigned k = 7;

    // matrix configurations of patches for GEMM
    // NOTE: HERE IS THE TEST DIFFERENCE BETWEEN single_gemm_xx.cpp files
    CBLAS_LAYOUT Layout = CblasColMajor;
    CBLAS_TRANSPOSE transa = CblasNoTrans;
    CBLAS_TRANSPOSE transb = CblasNoTrans;
    

    SingleGemmCudaDriver Test(Layout, transa, transb,
                              M, m, 
                              N, n, 
                              K, k);


    //Test.fillMemoryWith(3.0, 2.0, 1.0);
    Test.fillMemoryWithRandomNumbers();             
    SECTION("Compare GEMMs at the left-top corners (0,0) of matrices A, B, C") {

        SECTION("Changing ration: k > n > m") {
            k = 11;  n = 7; m = 4; 
            Test.changeSubPatchSizes(m, n, k);
            REQUIRE( Test.runTest() == true );
        }


        
        SECTION("Changing ration: n > m > k") {
            n = 21; m = 15; k = 8; 
            Test.changeSubPatchSizes(m, n, k);
            REQUIRE( Test.runTest() == true );
        }

        
        SECTION("Changing ration: m > k > n") {
            m = 18; k = 19; n = 9; 
            Test.changeSubPatchSizes(m, n, k);
            REQUIRE( Test.runTest() == true );
        }
        
    }

    
    //Test.fillMemoryWith(2.0, 2.0, 1.0);
    Test.fillMemoryWithRandomNumbers();
    SECTION("Compare GEMMs at arbitary starting addresses of matrices A, B, C") {

        // Starting addresses for patches (choose a diffenret points)
        Test.set_str_addressA(5, 7);
        Test.set_str_addressB(7, 8);
        Test.set_str_addressC(3, 4);

        SECTION("Changing ration: k > n > m") {
            k = 11;  n = 7; m = 4; 
            Test.changeSubPatchSizes(m, n, k);
            REQUIRE( Test.runTest() == true );
        }

        
        SECTION("Changing ration: n > m > k") {
            n = 21; m = 15; k = 8; 
            Test.changeSubPatchSizes(m, n, k);
            REQUIRE( Test.runTest() == true );
        }

        
        SECTION("Changing ration: m > k > n") {
            m = 18; k = 19; n = 9; 
            Test.changeSubPatchSizes(m, n, k);
            REQUIRE( Test.runTest() == true );
        }
    }


    //Test.fillMemoryWith(2.0, 2.0, 1.0);
    Test.fillMemoryWithRandomNumbers();
    SECTION("Compare GEMMs at right-down corners of matrices A, B, C") {

        SECTION("Changing ration: k > n > m") {
            k = 11;  n = 7; m = 4; 
            Test.changeSubPatchSizes(m, n, k);

            // Starting addresses for patches (choose a diffenret points)
            Test.set_str_addressA(M - m - 1, K - k - 1);
            Test.set_str_addressB(K - k - 1, N - n - 1);
            Test.set_str_addressC(M - m - 1, N - n - 1);

            REQUIRE( Test.runTest() == true );
        }

        
        SECTION("Changing ration: n > m > k") {
            n = 21; m = 15; k = 8; 
            Test.changeSubPatchSizes(m, n, k);

            // Starting addresses for patches (choose a diffenret points)
            Test.set_str_addressA(M - m - 1, K - k - 1);
            Test.set_str_addressB(K - k - 1, N - n - 1);
            Test.set_str_addressC(M - m - 1, N - n - 1);

            REQUIRE( Test.runTest() == true );
        }

        
        SECTION("Changing ration: m > k > n") {
            m = 18; k = 19; n = 9; 
            Test.changeSubPatchSizes(m, n, k);

            // Starting addresses for patches (choose a diffenret points)
            Test.set_str_addressA(M - m - 1, K - k - 1);
            Test.set_str_addressB(K - k - 1, N - n - 1);
            Test.set_str_addressC(M - m - 1, N - n - 1);

            REQUIRE( Test.runTest() == true );
        }
    }
}