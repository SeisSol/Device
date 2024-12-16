#include <cmath>
#include "algorithms/Common.h"

#include <omp.h>

namespace device {

template <typename T> T Algorithms::reduceVector(T *buffer, size_t size, const ReductionType type, void* streamPtr) {
    T value = deduceDefaultValue<T>(type);
    int* stream = reinterpret_cast<int*>(streamPtr);
    if (type == device::ReductionType::Add) {
        #pragma omp target teams distribute parallel for reduction(+:value) depend(inout: stream[0]) nowait device(TARGETDART_DEVICE(0))
        for (size_t i = 0; i < size; ++i) {
            value += buffer[i];
        }
    }
    if (type == device::ReductionType::Max) {
        value = 0;
        #pragma omp target teams distribute parallel for reduction(max:value) depend(inout: stream[0]) nowait device(TARGETDART_DEVICE(0))
        for (size_t i = 0; i < size; ++i) {
            value = std::max(value, buffer[i]);
        }
    }
    if (type == device::ReductionType::Min) {
        value = 0;
        #pragma omp target teams distribute parallel for reduction(min:value) depend(inout: stream[0]) nowait device(TARGETDART_DEVICE(0))
        for (size_t i = 0; i < size; ++i) {
            value = std::min(value, buffer[i]);
        }
    }
    #pragma omp taskwait depend(inout: stream[0])
    return value;
}

template unsigned Algorithms::reduceVector(unsigned *buffer, size_t size, ReductionType type, void* streamPtr);
template real Algorithms::reduceVector(real *buffer, size_t size, ReductionType type, void* streamPtr);

} // namespace device
