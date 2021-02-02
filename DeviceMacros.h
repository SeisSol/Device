#ifndef DEVICE_MACROS_H
#define DEVICE_MACROS_H

#if defined(DEVICE_CUDA_LANG)
#define DEVICE_KERNEL_LAUNCH(KRNL, GRID, BLOCK, SHR_SIZE, STREAM, ...) \
KRNL<<<GRID, BLOCK, SHR_SIZE, STREAM>>> ( __VA_ARGS__ )

#elif defined(DEVICE_HIP_LANG)
#define DEVICE_KERNEL_LAUNCH(KRNL, GRID, BLOCK, SHR_SIZE, STREAM, ...) \
hipLaunchKernelGGL(KRNL, GRID, BLOCK, SHR_SIZE, STREAM, __VA_ARGS__)

#else
#  error gpu interface not supported.
#endif

#endif //DEVICE_MACROS_H
