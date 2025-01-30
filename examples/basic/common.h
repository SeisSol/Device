// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_EXAMPLES_BASIC_COMMON_H_
#define SEISSOLDEVICE_EXAMPLES_BASIC_COMMON_H_

#include <cmath>
#include <iostream>

#if REAL_SIZE == 8
using real = double;
#define GEMM(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  cblas_dgemm(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#elif REAL_SIZE == 4
using real = float;
#define GEMM(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  cblas_sgemm(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
#error RealSize not supported.
#endif


#endif // SEISSOLDEVICE_EXAMPLES_BASIC_COMMON_H_

