#ifndef EXAMPLES_COMMON_H
#define EXAMPLES_COMMON_H

#include <cmath>
#include <iostream>

#if REAL_SIZE == 8
using real = double;
#define GEMM(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)                 \
  cblas_dgemm(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#elif REAL_SIZE == 4
using real = float;
#define GEMM(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)                 \
  cblas_sgemm(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
#error RealSize not supported.
#endif

#endif // EXAMPLES_COMMON_H
