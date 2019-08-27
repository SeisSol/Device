#include <iostream>
#include "device_utils.h"


#if REAL_SIZE == 8
    typedef double real;
#elif REAL_SIZE == 4
    typedef float real;
#else
#  error REAL_SIZE not supported.
#endif



int main() {

    // TODO: it must be a unit test!!!
    check_device_operation();

    size_t size = 10;
    double array_src[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double array_taget[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    for (int i = 0; i < 10; ++i) {
        std::cout << array_taget[i] << " ";
    }
    std::cout << std::endl;

    double *d_array =  (double*)device_malloc(size * sizeof(double));
    device_copy_to((void*)d_array, (void*)array_src, size * sizeof(double));
    device_copy_from((void*)array_taget, (void*)d_array, size * sizeof(double));
    device_free((void*)d_array);

    for (int i = 0; i < 10; ++i) {
        std::cout << array_taget[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}