#include <catch2/catch.hpp>

#include "copy_add_scale_driver.h"

TEST_CASE( "CopyScaleAdd", "[CopyScaleAdd]" ) {
    // global sizes of matrices
    const unsigned A_size[2] = {61, 53}; 
    const unsigned B_size[2] = {52, 29};
    const unsigned num_elements = 100;

    // patch sizes of the global matrices 
    unsigned m = 32;
    unsigned n = 9;

    // matrix configurations of patches for GEMM

    CopyAddScaleDriver Test(A_size,
                            B_size,
                            m, n,
                            num_elements);

    Test.fillMemoryWithRandomNumbers();             
    SECTION( "Compare copy scale add: alpha=1.0, beta=1.0" ) {
        Test.set_scales(1.0, 1.0);
        REQUIRE( Test.runTest() == true );
    }


    SECTION( "Compare copy scale add: alpha=2.0, beta=-1.0" ) {
        Test.set_scales(2.0, -1.0);
        REQUIRE( Test.runTest() == true );
    }

    SECTION( "Compare copy scale add: alpha=-2.0, beta=0.0" ) {
        Test.set_scales(-2.0, 0.0);
        REQUIRE( Test.runTest() == true );
    }
}