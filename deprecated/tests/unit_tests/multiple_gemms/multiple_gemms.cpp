#include <catch2/catch.hpp>

#include "multiple_gemm_driver.h"


TEST_CASE( "Multiple Gemm on CPU and GPU - A=CblasNoTrans, B=CblasNoTrans", "[GemmTest_NN]" ) {

    // set up sizes of tensors and compute tensors voluems
    const unsigned tensor_A_dims[3] = {51, 41, 37};
    const unsigned tensor_B_dims[3] = {41, 52, 39};
    const unsigned tensor_C_dims[3] = {45, 43, 39};
    const unsigned num_elements = 55;

    CBLAS_LAYOUT Layout = CblasColMajor;
    CBLAS_TRANSPOSE transa = CblasNoTrans;
    CBLAS_TRANSPOSE transb = CblasNoTrans;

    unsigned m = 11, n = 9, k = 10;

    MultipleGemmCudaDriver Test(Layout,
                                transa,
                                transb,
                                tensor_A_dims,
                                tensor_B_dims,
                                tensor_C_dims,
                                m, n, k,
                                num_elements);


    Test.fillMemoryWithRandomNumbers();

    SECTION("Compare GEMMs at the left-top corners (0,0) of matrices A, B, C") {
        Test.set_str_addressA(5, 7, 3);
        Test.set_str_addressB(7, 8, 10);
        Test.set_str_addressC(3, 4, 13);
        Test.set_leading_dimensions(tensor_A_dims[0],
                                    tensor_B_dims[0] * tensor_B_dims[1],
                                    tensor_C_dims[0]);

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
}



TEST_CASE( "Multiple Gemm on CPU and GPU - A=CblasTrans, B=CblasTrans", "[GemmTest_TT]" ) {

    // set up sizes of tensors and compute tensors voluems
    const unsigned tensor_A_dims[3] = {51, 41, 37};
    const unsigned tensor_B_dims[3] = {41, 52, 39};
    const unsigned tensor_C_dims[3] = {45, 43, 39};
    const unsigned num_elements = 55;

    CBLAS_LAYOUT Layout = CblasColMajor;
    CBLAS_TRANSPOSE transa = CblasTrans;
    CBLAS_TRANSPOSE transb = CblasTrans;

    unsigned m = 11, n = 9, k = 10;

    MultipleGemmCudaDriver Test(Layout,
                                transa,
                                transb,
                                tensor_A_dims,
                                tensor_B_dims,
                                tensor_C_dims,
                                m, n, k,
                                num_elements);


    Test.fillMemoryWithRandomNumbers();
    SECTION("Compare GEMMs at the left-top corners (0,0) of matrices A, B, C") {
        Test.set_str_addressA(5, 7, 3);
        Test.set_str_addressB(7, 8, 10);
        Test.set_str_addressC(3, 4, 13);
        Test.set_leading_dimensions(tensor_A_dims[0] * tensor_A_dims[0],
                                    tensor_B_dims[0] * tensor_B_dims[1],
                                    tensor_C_dims[0]);

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
}



TEST_CASE( "Multiple Gemm on CPU and GPU - A=CblasNoTrans, B=CblasTrans", "[GemmTest_NT]" ) {

    // set up sizes of tensors and compute tensors voluems
    const unsigned tensor_A_dims[3] = {51, 41, 37};
    const unsigned tensor_B_dims[3] = {41, 52, 39};
    const unsigned tensor_C_dims[3] = {45, 43, 39};
    const unsigned num_elements = 55;

    CBLAS_LAYOUT Layout = CblasColMajor;
    CBLAS_TRANSPOSE transa = CblasNoTrans;
    CBLAS_TRANSPOSE transb = CblasTrans;

    unsigned m = 11, n = 9, k = 10;

    MultipleGemmCudaDriver Test(Layout,
                                transa,
                                transb,
                                tensor_A_dims,
                                tensor_B_dims,
                                tensor_C_dims,
                                m, n, k,
                                num_elements);


    Test.fillMemoryWithRandomNumbers();
    SECTION("Compare GEMMs at the left-top corners (0,0) of matrices A, B, C") {
        Test.set_str_addressA(5, 7, 3);
        Test.set_str_addressB(7, 8, 10);
        Test.set_str_addressC(3, 4, 13);
        Test.set_leading_dimensions(tensor_A_dims[0] * tensor_A_dims[0],
                                    tensor_B_dims[0] * tensor_B_dims[1],
                                    tensor_C_dims[0] * tensor_C_dims[1]);

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
}


TEST_CASE( "Multiple Gemm on CPU and GPU - A=CblasTrans, B=CblasNoTrans", "[GemmTest_TN]" ) {

    // set up sizes of tensors and compute tensors voluems
    const unsigned tensor_A_dims[3] = {51, 41, 37};
    const unsigned tensor_B_dims[3] = {41, 52, 39};
    const unsigned tensor_C_dims[3] = {45, 43, 39};
    const unsigned num_elements = 55;

    CBLAS_LAYOUT Layout = CblasColMajor;
    CBLAS_TRANSPOSE transa = CblasTrans;
    CBLAS_TRANSPOSE transb = CblasNoTrans;

    unsigned m = 11, n = 9, k = 10;

    MultipleGemmCudaDriver Test(Layout,
                                transa,
                                transb,
                                tensor_A_dims,
                                tensor_B_dims,
                                tensor_C_dims,
                                m, n, k,
                                num_elements);


    Test.fillMemoryWithRandomNumbers();
    SECTION("Compare GEMMs at the left-top corners (0,0) of matrices A, B, C") {
        Test.set_str_addressA(5, 7, 3);
        Test.set_str_addressB(7, 8, 10);
        Test.set_str_addressC(3, 4, 13);
        Test.set_leading_dimensions(tensor_A_dims[0] * tensor_A_dims[0],
                                    tensor_B_dims[1],
                                    tensor_C_dims[0] * tensor_C_dims[1]);

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
}