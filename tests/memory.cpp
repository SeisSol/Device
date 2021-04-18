#include "BaseTestSuite.h"
#include "device.h"
#include "gtest/gtest.h"
#include <functional>
#include <numeric>
#include <vector>

using namespace device;
using namespace ::testing;

class Memories : public BaseTestSuite {
  using BaseTestSuite::BaseTestSuite;
};

TEST_F(Memories, copy2DMemory) {
  const int M = 150;
  const int N = 100;

  int hostVector[M][N];

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      hostVector[i][j] = 1904;
    }
  }

  int *arr = (int *)device->api->allocGlobMem(M * N * sizeof(int));

  int spitch = N * sizeof(int);
  int dpitch = N * sizeof(int);
  int width = N * sizeof(int);
  int height = M;

  device->api->copy2dArrayTo(arr, dpitch, &hostVector[0], spitch, width, height);
  int hostVector2[M][N];
  device->api->copy2dArrayFrom(&hostVector2[0], dpitch, arr, spitch, width, height);

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      EXPECT_EQ(1904, hostVector2[i][j]);
    }
  }

  device->api->freeMem(arr);
}

TEST_F(Memories, copy2DMemoryWithSrcPitch) {

  const int SPI = 2;
  const int M = 150;
  const int N = 100;

  int hostVector[M][N + SPI];

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N + SPI; j++) {
      hostVector[i][j] = 1904;
    }
  }

  int *arr = (int *)device->api->allocGlobMem(M * N * sizeof(int));

  int spitch = (N + SPI) * sizeof(int);
  int dpitch = N * sizeof(int);
  int width = N * sizeof(int);
  int height = M;

  device->api->copy2dArrayTo(arr, dpitch, &hostVector[0], spitch, width, height);
  int hostVector2[M][N];
  device->api->copy2dArrayFrom(&hostVector2[0], dpitch, arr, dpitch, width, height);

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      EXPECT_EQ(1904, hostVector2[i][j]);
    }
  }

  device->api->freeMem(arr);
}

TEST_F(Memories, copy2DMemoryWithDstPitch) {

  const int SPI = 2;
  const int DPI = 4;

  const int M = 150;
  const int N = 100;

  int hostVector[M][N + SPI];

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N + SPI; j++) {
      hostVector[i][j] = 1904;
    }
  }

  int *arr = (int *)device->api->allocGlobMem(M * (N + DPI) * sizeof(int));

  int spitch = (N + SPI) * sizeof(int);
  int dpitch = (N + DPI) * sizeof(int);
  int width = N * sizeof(int);
  int height = M;

  device->api->copy2dArrayTo(arr, dpitch, &hostVector[0], spitch, width, height);
  int hostVector2[M][N + SPI];
  device->api->copy2dArrayFrom(&hostVector2[0], spitch, arr, dpitch, width, height);

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N + SPI; j++) {
      EXPECT_EQ(1904, hostVector2[i][j]);
    }
  }

  device->api->freeMem(arr);
}