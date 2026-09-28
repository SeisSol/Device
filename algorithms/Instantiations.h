// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_ALGORITHMS_INSTANTIATIONS_H_
#define SEISSOLDEVICE_ALGORITHMS_INSTANTIATIONS_H_

// The types for which every backend provides the member templates of device::Algorithms.
//
// Each list applies the macro given as its argument to every entry. A backend instantiates a
// member template by passing the matching DEVICE_ALGORITHMS_INSTANTIATE_* macro to its list,
// inside of namespace device; e.g.
//
//   DEVICE_ALGORITHMS_ARRAY_TYPES(DEVICE_ALGORITHMS_INSTANTIATE_SCALE_ARRAY)
//
// Thus, all backends export the same symbols, and a consumer that links against one of them
// links against all of them.

// scaleArray, fillArray
#define DEVICE_ALGORITHMS_ARRAY_TYPES(X) X(float) X(double) X(int) X(unsigned) X(char)

// setToValue
#define DEVICE_ALGORITHMS_VALUE_TYPES(X) DEVICE_ALGORITHMS_ARRAY_TYPES(X) X(long) X(unsigned long)

// accumulateBatchedData, compareDataWithHost
#define DEVICE_ALGORITHMS_FLOATING_TYPES(X) X(float) X(double)

// reduceVector; each entry is (accumulator type, vector element type)
#define DEVICE_ALGORITHMS_REDUCTION_TYPES(X)                                                       \
  X(int, int)                                                                                      \
  X(unsigned, unsigned)                                                                            \
  X(long, int)                                                                                     \
  X(unsigned long, unsigned)                                                                       \
  X(long, long)                                                                                    \
  X(unsigned long, unsigned long)                                                                  \
  X(long long, int)                                                                                \
  X(unsigned long long, unsigned)                                                                  \
  X(long long, long)                                                                               \
  X(unsigned long long, unsigned long)                                                             \
  X(long long, long long)                                                                          \
  X(unsigned long long, unsigned long long)                                                        \
  X(float, float)                                                                                  \
  X(double, float)                                                                                 \
  X(double, double)

#define DEVICE_ALGORITHMS_INSTANTIATE_SCALE_ARRAY(T)                                               \
  template void Algorithms::scaleArray<T>(T*, T, size_t, void*);

#define DEVICE_ALGORITHMS_INSTANTIATE_FILL_ARRAY(T)                                                \
  template void Algorithms::fillArray<T>(T*, T, size_t, void*);

#define DEVICE_ALGORITHMS_INSTANTIATE_SET_TO_VALUE(T)                                              \
  template void Algorithms::setToValue<T>(T**, T, size_t, size_t, void*);

#define DEVICE_ALGORITHMS_INSTANTIATE_ACCUMULATE_BATCHED_DATA(T)                                   \
  template void Algorithms::accumulateBatchedData<T>(const T**, T**, size_t, size_t, void*);

#define DEVICE_ALGORITHMS_INSTANTIATE_COMPARE_DATA_WITH_HOST(T)                                    \
  template void Algorithms::compareDataWithHost<T>(const T*, const T*, size_t, const std::string&);

#define DEVICE_ALGORITHMS_INSTANTIATE_REDUCE_VECTOR(AccT, VecT)                                    \
  template void Algorithms::reduceVector<AccT, VecT>(                                              \
      AccT*, const VecT*, bool, size_t, ReductionType, void*);

#endif // SEISSOLDEVICE_ALGORITHMS_INSTANTIATIONS_H_
