#ifndef DEVICE_MACROS_H
#define DEVICE_MACROS_H

#if defined(CUDA)
#define DEVICE_KERNEL_LAUNCH(KRNL, GRID, SHR_SIZE, STREAM, BLOCK, ...) \
KRNL<<<GRID, BLOCK, SHR_SIZE, STREAM>>>(##__VA_ARGS__)

#elif defined(HIP)
#define DEVICE_KERNEL_LAUNCH(KRNL, GRID, SHR_SIZE, STREAM, BLOCK, ...) \
hipLaunchKernelGGL(KRNL, GRID, BLOCK, SHR_SIZE, STREAM, ##__VA_ARGS__)

#else
#  error gpu interface not supported.
#endif

#endif //DEVICE_MACROS_H
