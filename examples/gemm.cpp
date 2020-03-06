#include "Common.h"
#include "device.h"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <vector>

using namespace device;

int main(int Argc, char *Argv[]) {

  // define parameters for a batched gemm
  const unsigned NUM_ELEMENTS = 100;
  int M = 56, N = 9, K = 9;
  real Alpha = 1.0, Beta = 1.0;

  // compute sizes of matrices
  const unsigned C_SIZE = M * N;
  const unsigned A_SIZE = M * K;
  const unsigned B_SIZE = K * N;

  // allocate data
  real *C = new real[C_SIZE * NUM_ELEMENTS];
  real *A = new real[A_SIZE * NUM_ELEMENTS];
  real *B = new real[B_SIZE * NUM_ELEMENTS];

  // Init matrices
  for (unsigned ElementIdx = 0; ElementIdx < NUM_ELEMENTS; ++ElementIdx) {
    for (int I = 0; I < C_SIZE; ++I) {
      C[I + ElementIdx * C_SIZE] = getRandom();
    }

    for (int I = 0; I < A_SIZE; ++I) {
      A[I + ElementIdx * A_SIZE] = getRandom();
    }

    for (int I = 0; I < B_SIZE; ++I) {
      B[I + ElementIdx * B_SIZE] = getRandom();
    }
  }

  DeviceInstance &Device = DeviceInstance::getInstance();

  // allocate data on a device
  real *DC = static_cast<real *>(Device.api->allocGlobMem(C_SIZE * NUM_ELEMENTS * sizeof(real)));
  real *DA = static_cast<real *>(Device.api->allocGlobMem(A_SIZE * NUM_ELEMENTS * sizeof(real)));
  real *DB = static_cast<real *>(Device.api->allocGlobMem(B_SIZE * NUM_ELEMENTS * sizeof(real)));

  // copy data into a device
  Device.api->copyTo(DC, C, C_SIZE * NUM_ELEMENTS * sizeof(real));
  Device.api->copyTo(DA, A, A_SIZE * NUM_ELEMENTS * sizeof(real));
  Device.api->copyTo(DB, B, B_SIZE * NUM_ELEMENTS * sizeof(real));

  // run GEMM on CPU
  for (unsigned ElementIdx = 0; ElementIdx < NUM_ELEMENTS; ++ElementIdx) {
    real *NextC = &C[ElementIdx * C_SIZE];
    real *NextA = &A[ElementIdx * A_SIZE];
    real *NextB = &B[ElementIdx * B_SIZE];
    GEMM(CblasColMajor,
         CblasNoTrans,
         CblasNoTrans,
         M,
         N,
         K,
         Alpha,
         NextA,
         M,
         NextB,
         K,
         Beta,
         NextC,
         M);
  }

  // run GEMM on GPU using arrays of pointers as the addressing mode
  // 1. shuffle pointers
  std::vector<real *> PtrC{};
  std::vector<real *> PtrA{};
  std::vector<real *> PtrB{};
  std::vector<unsigned> Indices(NUM_ELEMENTS, 0);
  unsigned Counter = 0;
  std::for_each(Indices.begin(), Indices.end(), [&Counter](unsigned &Index) { Index = Counter++; });
  std::random_shuffle(Indices.begin(), Indices.end());

  for (auto Index : Indices) {
    PtrC.push_back(&DC[Index * C_SIZE]);
    PtrA.push_back(&DA[Index * A_SIZE]);
    PtrB.push_back(&DB[Index * B_SIZE]);
  }

  // allocate pointers on device. use stack memory to store indices (just for testing)
  real **DPtrC =
      reinterpret_cast<real **>(Device.api->getStackMemory(NUM_ELEMENTS * sizeof(real *)));
  real **DPtrA =
      reinterpret_cast<real **>(Device.api->getStackMemory(NUM_ELEMENTS * sizeof(real *)));
  real **DPtrB =
      reinterpret_cast<real **>(Device.api->getStackMemory(NUM_ELEMENTS * sizeof(real *)));

  // move pointers to GPU
  Device.api->copyTo(DPtrC, PtrC.data(), NUM_ELEMENTS * sizeof(real *));
  Device.api->copyTo(DPtrA, PtrA.data(), NUM_ELEMENTS * sizeof(real *));
  Device.api->copyTo(DPtrB, PtrB.data(), NUM_ELEMENTS * sizeof(real *));

  // call device gemm
  Device.gemm(ColMajor,
              NoTrans,
              NoTrans,
              M,
              N,
              K,
              Alpha,
              DPtrA,
              M,
              DPtrB,
              K,
              Beta,
              DPtrC,
              M,
              0,
              0,
              0,
              NUM_ELEMENTS);

  // compare results
  real *DeviceResults = new real[C_SIZE * NUM_ELEMENTS];
  Device.api->copyFrom(DeviceResults, DC, C_SIZE * NUM_ELEMENTS * sizeof(real));
  compareResults(DeviceResults, C, C_SIZE, NUM_ELEMENTS);

  // deallocate data from a device
  Device.api->freeMem(DC);
  Device.api->freeMem(DA);
  Device.api->freeMem(DB);

  Device.api->popStackMemory();
  Device.api->popStackMemory();
  Device.api->popStackMemory();

  // print report from device
  std::cout << Device.api->getMemLeaksReport() << std::endl;

  // deallocate data from the host
  delete[] C;
  delete[] A;
  delete[] B;
  delete[] DeviceResults;

  Device.finalize();
}