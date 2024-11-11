// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_DEVICEMACROS_H_
#define SEISSOLDEVICE_DEVICEMACROS_H_

#if defined(DEVICE_CUDA_LANG)
#define DEVICE_KERNEL_LAUNCH(KRNL, GRID, BLOCK, SHR_SIZE, STREAM, ...) \
  KRNL<<<GRID, BLOCK, SHR_SIZE, STREAM>>>(__VA_ARGS__)

#elif defined(DEVICE_HIP_LANG)
#define DEVICE_KERNEL_LAUNCH(KRNL, GRID, BLOCK, SHR_SIZE, STREAM, ...) \
  hipLaunchKernelGGL(KRNL, GRID, BLOCK, SHR_SIZE, STREAM, __VA_ARGS__)

#elif defined(DEVICE_ONEAPI_LANG)
#error do not launch kernel with macros using OneAPI

#elif defined(DEVICE_HIPSYCL_LANG)
#error do not launch kernel with macros using hipsycl

#else
#error gpu interface not supported.
#endif


#endif // SEISSOLDEVICE_DEVICEMACROS_H_

