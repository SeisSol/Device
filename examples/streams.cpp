#include <iostream>
#include <array>
#include <vector>
#include <cblas.h>

#include <algorithm>
#include "device.h"
#include "Common.h"


using namespace device;

int main(int argc, char *argv[]) {

  // define a problem size
  const unsigned NumStreams = 8;
  const unsigned NumElementsPerStream = 1024 * 16;
  const unsigned NumElements = NumStreams * NumElementsPerStream;

  // allocate streams
  Device& device = Device::getInstance();
  std::array<int, NumStreams> StreamIds;
  std::for_each(StreamIds.begin(), StreamIds.end(), [&device](int &StreamId) {StreamId = device.api->createStream();});



  // define parameters for a batched gemm
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

  /*
  device.gemm(ColMajor, NoTrans, NoTrans,
              m, n, k,
              Alpha, d_PtrA, m,
              d_PtrB, k,
              Beta, d_PtrC, m,
              0, 0, 0, NumElements);
  */
  // call device gemm
  for (unsigned StreamCounter = 0; StreamCounter < NumStreams; ++StreamCounter) {
    device.api->setComputeStream(StreamIds[StreamCounter]);
    device.gemm(ColMajor, NoTrans, NoTrans,
                m, n, k,
                Alpha, &d_PtrA[StreamCounter * NumElementsPerStream], m,
                &d_PtrB[StreamCounter * NumElementsPerStream], k,
                Beta, &d_PtrC[StreamCounter * NumElementsPerStream], m,
                0, 0, 0, NumElementsPerStream);

  }
  device.api->setDefaultComputeStream();

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

  // delete streams
  std::for_each(StreamIds.begin(), StreamIds.end(), [&device](int &StreamId) {device.api->deleteStream(StreamId);});

  // deallocate data from the host
  delete [] C;
  delete [] A;
  delete [] B;
  delete [] DeviceResults;

  device.finalize();
}