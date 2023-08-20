#ifndef DEVICE_MACROS_H
#define DEVICE_MACROS_H

#if defined(DEVICE_CUDA_LANG)
#define DEVICE_KERNEL_LAUNCH(KRNL, GRID, BLOCK, SHR_SIZE, STREAM, ...) \
  KRNL<<<GRID, BLOCK, SHR_SIZE, STREAM>>>(__VA_ARGS__)

#elif defined(DEVICE_HIP_LANG)
#define DEVICE_KERNEL_LAUNCH(KRNL, GRID, BLOCK, SHR_SIZE, STREAM, ...) \
  hipLaunchKernelGGL(KRNL, GRID, BLOCK, SHR_SIZE, STREAM, __VA_ARGS__)

#elif defined(DEVICE_ONEAPI_LANG)
#error do not launch kernel with macros using OneAPI

#elif defined(DEVICE_OPENSYCL_LANG)
#error do not launch kernel with macros using opensycl

#else
#error gpu interface not supported.
#endif

#endif // DEVICE_MACROS_H
