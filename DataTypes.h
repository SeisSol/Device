#ifndef DEVICE_DATA_TYPES_H
#define DEVICE_DATA_TYPES_H


#if REAL_SIZE == 8
typedef double real;
#elif REAL_SIZE == 4
typedef float real;
#else
#  error REAL_SIZE not supported.
#endif

#endif // DEVICE_DATA_TYPES_H