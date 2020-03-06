#ifndef EXAMPLES_COMMON_H
#define EXAMPLES_COMMON_H

#include <cmath>
#include <iostream>

#if REAL_SIZE == 8
#define GEMM(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)                 \
  cblas_dgemm(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#elif REAL_SIZE == 4
#define GEMM(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)                 \
  cblas_sgemm(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
#error RealSize not supported.
#endif

real getRandom() { return static_cast<real>(std::rand()) / RAND_MAX; }

void compareResults(const real *DeviceSolution, const real *ReferenceSolution,
                    const unsigned ElementSize, const unsigned NumElements) {
  bool Equal = true;
  real MaxAbsDiff = 0.0;
  const real Esp = 1e-8;
  for (unsigned e = 0; e < NumElements; ++e) {
    const real *NextDeviceElement = &DeviceSolution[e * ElementSize];
    const real *NextReferenceElement = &ReferenceSolution[e * ElementSize];

    for (int i = 0; i < ElementSize; ++i) {
      real Diff = std::fabs(NextReferenceElement[i] - NextDeviceElement[i]);
      if (Diff > Esp) {
        Equal = false;
      }
      MaxAbsDiff = std::max(MaxAbsDiff, Diff);
    }
  }

  if (!Equal) {
    std::cout << "::"
              << "ERROR:: Results are nor equal. Max diff = " << MaxAbsDiff << std::endl;
  } else {
    std::cout << "::"
              << "Results are correct. Max diff = " << MaxAbsDiff << std::endl;
  }
}

#endif // EXAMPLES_COMMON_H
