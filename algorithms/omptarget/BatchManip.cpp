#include "AbstractAPI.h"
//#include "interfaces/omptarget/Internals.h"
#include <device.h>
#include <cassert>

namespace device {

  void Algorithms::streamBatchedData(real **baseSrcPtr,
                                     real **baseDstPtr,
                                     unsigned elementSize,
                                     unsigned numElements,
                                     void* streamPtr) {
    int* stream = reinterpret_cast<int*>(streamPtr);
    #pragma omp target teams distribute depend(inout: stream[0]) nowait
    for (size_t i = 0; i < numElements; ++i) {
        #pragma omp parallel
        {
            real *srcElement = baseSrcPtr[i];
            real *dstElement = baseDstPtr[i];
            #pragma omp for
            for (size_t j = 0; j < elementSize; ++j) {
                dstElement[j] = srcElement[j];
            }
        }
    }
  }

  void Algorithms::accumulateBatchedData(real **baseSrcPtr,
                                         real **baseDstPtr,
                                         unsigned elementSize,
                                         unsigned numElements,
                                         void* streamPtr) {
    int* stream = reinterpret_cast<int*>(streamPtr);
    #pragma omp target teams distribute depend(inout: stream[0]) nowait
    for (size_t i = 0; i < numElements; ++i) {
        #pragma omp parallel
        {
            real *srcElement = baseSrcPtr[i];
            real *dstElement = baseDstPtr[i];
            #pragma omp for
            for (size_t j = 0; j < elementSize; ++j) {
                dstElement[j] += srcElement[j];
            }
        }
    }
  }

  void Algorithms::touchBatchedMemory(real **basePtr,
                                      unsigned elementSize,
                                      unsigned numElements,
                                      bool clean,
                                      void* streamPtr) {
    int* stream = reinterpret_cast<int*>(streamPtr);
    #pragma omp target teams distribute depend(inout: stream[0]) nowait
    for (size_t i = 0; i < numElements; ++i) {
        #pragma omp parallel
        {
            real *element = basePtr[i];
            if (element != nullptr) {
                if (clean) {
                    #pragma omp for
                    for (size_t j = 0; j < elementSize; ++j) {
                        element[j] = 0;
                    }
                }
                else {
                    #pragma omp for
                    for (size_t j = 0; j < elementSize; ++j) {
                        real& value = element[j];
                        // Do something dummy here. We just need to check the pointers point to valid memory locations.
                        // Avoid compiler optimization. Possibly, implement a dummy code with asm.
                        value += 1.0;
                        value -= 1.0;
                    }
                }
            }
        }
    }
  }

void Algorithms::setToValue(real** out, real value, size_t elementSize, size_t numElements, void* streamPtr) {
  int* stream = reinterpret_cast<int*>(streamPtr);
    #pragma omp target teams distribute depend(inout: stream[0]) nowait
    for (size_t i = 0; i < numElements; ++i) {
        #pragma omp parallel
        {
            real *element = out[i];
            #pragma omp for
            for (size_t j = 0; j < elementSize; ++j) {
                element[j] = value;
            }
        }
    }
}

  template<typename T>
  void Algorithms::copyUniformToScatter(T *src,
                                        T **dst,
                                        size_t srcOffset,
                                        size_t copySize,
                                        size_t numElements,
                                        void* streamPtr) {
                                            int* stream = reinterpret_cast<int*>(streamPtr);
    #pragma omp target teams distribute depend(inout: stream[0]) nowait
    for (size_t i = 0; i < numElements; ++i) {
        #pragma omp parallel
        {
            T *srcElement = &src[i * srcOffset];
            T *dstElement = dst[i];
            #pragma omp for
            for (size_t j = 0; j < copySize; ++j) {
                dstElement[j] = srcElement[j];
            }
        }
    }
  }
  template void Algorithms::copyUniformToScatter(real *src,
                                                 real **dst,
                                                 size_t srcOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);

  template void Algorithms::copyUniformToScatter(int *src,
                                                 int **dst,
                                                 size_t srcOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);

  template void Algorithms::copyUniformToScatter(char *src,
                                                 char **dst,
                                                 size_t srcOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);

  template<typename T>
  void Algorithms::copyScatterToUniform(T **src,
                                        T *dst,
                                        size_t dstOffset,
                                        size_t copySize,
                                        size_t numElements,
                                        void* streamPtr) {
                                            int* stream = reinterpret_cast<int*>(streamPtr);
    #pragma omp target teams distribute depend(inout: stream[0]) nowait
    for (size_t i = 0; i < numElements; ++i) {
        #pragma omp parallel
        {
            T *srcElement = src[i];
            T *dstElement = &dst[i * dstOffset];
            #pragma omp for
            for (size_t j = 0; j < copySize; ++j) {
                dstElement[j] = srcElement[j];
            }
        }
    }
  }
  template void Algorithms::copyScatterToUniform(real **src,
                                                 real *dst,
                                                 size_t dstOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);

  template void Algorithms::copyScatterToUniform(int **src,
                                                 int *dst,
                                                 size_t dstOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);

  template void Algorithms::copyScatterToUniform(char **src,
                                                 char *dst,
                                                 size_t dstOffset,
                                                 size_t copySize,
                                                 size_t numElements,
                                                 void* streamPtr);
} // namespace device
