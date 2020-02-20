#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#include <cblas.h>
#include "device.h"
using namespace device;

#if REAL_SIZE == 8
#define GEMM(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  cblas_dgemm(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#elif REAL_SIZE == 4
#define GEMM(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  cblas_sgemm(LAYOUT, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
#  error RealSize not supported.
#endif



real getRandom() {
  return static_cast<real>(std::rand()) / RAND_MAX;
}

void compareResults(const real* DeviceSolution,
                    const real* ReferenceSolution,
                    const unsigned ElementSize,
                    const unsigned NumElements) {
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
    std::cout << "::" << "ERROR:: Results are nor equal. Max diff = " << MaxAbsDiff << std::endl;
  }
  else {
    std::cout << "::" << "Results are correct. Max diff = " << MaxAbsDiff << std::endl;
  }
}


int main() {

  // define parameters for a batched gemm
  const unsigned NumElements = 100;
  int m = 56, n = 9, k = 9;
  real Alpha = 1.0, Beta = 1.0;

  // compute sizes of matrices
  const unsigned C_SIZE = m * n;
  const unsigned A_SIZE = m * k;
  const unsigned B_SIZE = k * n;

  // allocate data
  real *C = new real[C_SIZE * NumElements];
  real *A = new real[A_SIZE * NumElements];
  real *B = new real[B_SIZE * NumElements];

  // Init matrices
  for (unsigned e = 0; e < NumElements; ++e) {
    for (int i = 0; i < C_SIZE; ++i) {
      C[i + e * C_SIZE] = getRandom();
    }

    for (int i = 0; i < A_SIZE; ++ i) {
      A[i + e * A_SIZE] = getRandom();
    }

    for (int i = 0; i < B_SIZE; ++ i) {
      B[i + e * B_SIZE] = getRandom();
    }
  }

  Device& device = Device::getInstance();

  // allocate data on a device
  real *d_C = static_cast<real*>(device.api->allocGlobMem(C_SIZE * NumElements * sizeof(real)));
  real *d_A = static_cast<real*>(device.api->allocGlobMem(A_SIZE * NumElements * sizeof(real)));
  real *d_B = static_cast<real*>(device.api->allocGlobMem(B_SIZE * NumElements * sizeof(real)));

  // copy data into a device
  device.api->copyTo(d_C, C, C_SIZE * NumElements * sizeof(real));
  device.api->copyTo(d_A, A, A_SIZE * NumElements * sizeof(real));
  device.api->copyTo(d_B, B, B_SIZE * NumElements * sizeof(real));

  // run GEMM on CPU
  for (unsigned e = 0; e < NumElements; ++e) {
    real *next_C = &C[e * C_SIZE];
    real *next_A = &A[e * A_SIZE];
    real *next_B = &B[e * B_SIZE];
    GEMM(CblasColMajor, CblasNoTrans, CblasNoTrans,
         m, n, k,
         Alpha, next_A, m,
         next_B, k,
         Beta, next_C, m);
  }

  // run GEMM on GPU using arrays of pointers as the addressing mode
  // 1. shuffle pointers
  std::vector<real*> PtrC{};
  std::vector<real*> PtrA{};
  std::vector<real*> PtrB{};
  std::vector<unsigned> Indices(NumElements, 0);
  unsigned Counter = 0;
  std::for_each(Indices.begin(), Indices.end(), [&Counter](unsigned &Index) {Index = Counter++;});
  std::random_shuffle(Indices.begin(), Indices.end());

  for (auto Index: Indices) {
    PtrC.push_back(&d_C[Index * C_SIZE]);
    PtrA.push_back(&d_A[Index * A_SIZE]);
    PtrB.push_back(&d_B[Index * B_SIZE]);
  }

  // allocate pointers on device. use stack memory to store indices (just for testing)
  real **d_PtrC = reinterpret_cast<real**>(device.api->getStackMemory(NumElements * sizeof(real*)));
  real **d_PtrA = reinterpret_cast<real**>(device.api->getStackMemory(NumElements * sizeof(real*)));
  real **d_PtrB = reinterpret_cast<real**>(device.api->getStackMemory(NumElements * sizeof(real*)));

  // move pointers to GPU
  device.api->copyTo(d_PtrC, PtrC.data(), NumElements * sizeof(real*));
  device.api->copyTo(d_PtrA, PtrA.data(), NumElements * sizeof(real*));
  device.api->copyTo(d_PtrB, PtrB.data(), NumElements * sizeof(real*));

  // call device gemm
  device.gemm(ColMajor, NoTrans, NoTrans,
              m, n, k,
              Alpha, d_PtrA, m,
              d_PtrB, k,
              Beta, d_PtrC, m,
              0, 0, 0, NumElements);

  // compare results
  real *DeviceResults = new real[C_SIZE * NumElements];
  device.api->copyFrom(DeviceResults, d_C, C_SIZE * NumElements * sizeof(real));
  compareResults(DeviceResults, C, C_SIZE, NumElements);

  // deallocate data from a device
  device.api->freeMem(d_C);
  device.api->freeMem(d_A);
  device.api->freeMem(d_B);

  device.api->popStackMemory();
  device.api->popStackMemory();
  device.api->popStackMemory();

  // print report from device
  std::cout << device.api->getMemLeaksReport() << std::endl;

  // deallocate data from the host
  delete [] C;
  delete [] A;
  delete [] B;
  delete [] DeviceResults;

  device.finalize();
}