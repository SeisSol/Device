#include <stdlib.h>
#include <assert.h>
#include <bitset>
#include "helper.h"

void fillWithStuff(real* matrix, const unsigned arr_size) {
  for (unsigned j = 0; j < arr_size; ++j) {
      matrix[j] = drand48();
  }
}


void fillWithStuff(real* matrix, unsigned arr_size, const real defualt_value) {
  for (unsigned j = 0; j < arr_size; ++j) {
      matrix[j] = defualt_value;
  }
}


union {
    real input;
    bytes output;
} number;

const unsigned count_incorrect_bits(real ref, real computed) {

    assert((sizeof(real) == sizeof(bytes)) && "wrong mapping b/w of real and its bytes");

    number.input = ref;
    std::bitset<sizeof(bytes) * 8> ref_bits(number.output);

    number.input = computed;
    std::bitset<sizeof(bytes) * 8> computed_bits(number.output);


    int num_correct_bits = 0;
    for(int i = ref_bits.size() - 1; i >= 0; --i) {
        if (ref_bits[i] == computed_bits[i]) 
            ++num_correct_bits;
        else 
            break;
    }
    const unsigned num_wrong_bits = ref_bits.size() - num_correct_bits;
    return num_wrong_bits;
}