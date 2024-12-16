#include "AbstractAPI.h"
//#include "interfaces/omptarget/Internals.h"
#include <cassert>
#include <device.h>

#include <omp.h>

namespace device {
template <typename T> void Algorithms::scaleArray(T *devArray,
                                                  T scalar,
                                                  const size_t numElements,
                                                  void* streamPtr) {
                                                    int* stream = reinterpret_cast<int*>(streamPtr);
  #pragma omp target teams distribute parallel for depend(inout: stream[0]) nowait device(TARGETDART_DEVICE(0))
    for (size_t i = 0; i < numElements; ++i) {
            devArray[i] *= scalar;
    }
}
template void Algorithms::scaleArray(real *devArray, real scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(int *devArray, int scalar, const size_t numElements, void* streamPtr);
template void Algorithms::scaleArray(char *devArray, char scalar, const size_t numElements, void* streamPtr);


template <typename T> void Algorithms::fillArray(T *devArray, const T scalar, const size_t numElements, void* streamPtr) {
    int* stream = reinterpret_cast<int*>(streamPtr);
  #pragma omp target teams distribute parallel for depend(inout: stream[0]) nowait device(TARGETDART_DEVICE(0))
    for (size_t i = 0; i < numElements; ++i) {
            devArray[i] = scalar;
    }
}
template void Algorithms::fillArray(real *devArray, real scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(int *devArray, int scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(unsigned *devArray, unsigned scalar, const size_t numElements, void* streamPtr);
template void Algorithms::fillArray(char *devArray, char scalar, const size_t numElements, void* streamPtr);

void Algorithms::touchMemory(real *ptr, size_t size, bool clean, void* streamPtr) {
    int* stream = reinterpret_cast<int*>(streamPtr);
  #pragma omp target teams distribute parallel for depend(inout: stream[0]) nowait device(TARGETDART_DEVICE(0))
    for (size_t i = 0; i < size; ++i) {
            if (clean) {
            ptr[i] = 0;
            } else {
            real& value = ptr[i];
            // Do something dummy here. We just need to check the pointers point to valid memory locations.
            // Avoid compiler optimization. Possibly, implement a dummy code with asm.
            value += 1;
            value -= 1;
            }
    }
}


void Algorithms::incrementalAdd(
  real** out,
  real *base,
  size_t increment,
  size_t numElements,
  void* streamPtr) {
int* stream = reinterpret_cast<int*>(streamPtr);
   #pragma omp target teams distribute parallel for depend(inout: stream[0]) nowait device(TARGETDART_DEVICE(0))
    for (size_t i = 0; i < numElements; ++i) {
        out[i] = base + i * increment;
    }
}
} // namespace device
