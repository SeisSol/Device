#include "device.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace device;
using namespace ::testing;

class Reductions : public ::testing::Test {
public:
  static DeviceInstance *device;

  Reductions() { randomEngine.seed(randomDevice()); }

  static void SetUpTestSuite() {
    device = &DeviceInstance::getInstance();
    device->api->allocateStackMem();
  }

  static void TearDownTestSuite() { device->finalize(); }

protected:
  std::random_device randomDevice;
  std::mt19937 randomEngine;
};

DeviceInstance *Reductions::device = nullptr;

TEST_F(Reductions, Add) {
  constexpr size_t size = 1000;
  std::vector<int> vector(size, 0);

  std::uniform_int_distribution<> distribution(-50, 50);
  for (auto &element : vector) {
    element = distribution(randomEngine);
  }

  int *devVector = reinterpret_cast<int *>(device->api->getStackMemory(sizeof(int) * size));
  device->api->copyTo(devVector, vector.data(), sizeof(int) * size);

  int expectedResult = std::accumulate(vector.begin(), vector.end(), 0, std::plus<int>());

  int testResult = device->algorithms.reduceVector(devVector, size, ReductionType::Add);
  device->api->popStackMemory();
  EXPECT_EQ(expectedResult, testResult);
}

TEST_F(Reductions, Max) {
  constexpr size_t size = 1000;
  std::vector<int> vector(size, 0);

  int *devVector = reinterpret_cast<int *>(device->api->getStackMemory(sizeof(int) * size));
  device->api->copyTo(devVector, vector.data(), sizeof(int) * size);

  std::uniform_int_distribution<> distribution(-50, 50);
  for (auto &element : vector) {
    element = distribution(randomEngine);
  }

  device->api->copyTo(devVector, vector.data(), sizeof(int) * size);

  auto max = [](int a, int b) -> int { return a > b ? a : b; };
  int expectedResult = std::accumulate(vector.begin(), vector.end(), 0, max);

  int testResult = device->algorithms.reduceVector(devVector, size, ReductionType::Max);
  device->api->popStackMemory();
  EXPECT_EQ(expectedResult, testResult);
}

TEST_F(Reductions, MIN) {
  constexpr size_t size = 1000;
  std::vector<int> vector(size, 0);

  std::uniform_int_distribution<> distribution(-50, 50);
  for (auto &element : vector) {
    element = distribution(randomEngine);
  }

  int *devVector = reinterpret_cast<int *>(device->api->getStackMemory(sizeof(int) * size));
  device->api->copyTo(devVector, vector.data(), sizeof(int) * size);

  auto min = [](int a, int b) -> int { return a > b ? b : a; };
  int expectedResult = std::accumulate(vector.begin(), vector.end(), 0, min);

  int testResult = device->algorithms.reduceVector(devVector, size, ReductionType::Min);
  device->api->popStackMemory();
  EXPECT_EQ(expectedResult, testResult);
}

TEST_F(Reductions, fill) {

  const int N = 100;
  int *arr = (int *)device->api->allocGlobMem(N * sizeof(int));
  int scalar = 502;

  device->algorithms.fillArray(arr, scalar, N);

  std::vector<int> hostVector(N, 0);
  device->api->copyFrom(&hostVector[0], arr, N * sizeof(int));

  for (auto &i : hostVector) {
    EXPECT_EQ(scalar, i);
  }

  device->api->freeMem(arr);
}

TEST_F(Reductions, touchClean) {

  const int N = 100;
  real *arr = (real *)device->api->allocGlobMem(N * sizeof(real));
  device->algorithms.touchMemory(arr, N, true);
  std::vector<real> hostVector(N, 0);

  device->api->copyFrom(&hostVector[0], arr, N * sizeof(real));

  for (auto &i : hostVector) {
    EXPECT_EQ(0, i);
  }

  device->api->freeMem(arr);
}

TEST_F(Reductions, touchNoClean) {

  const int N = 100;
  real *arr = (real *)device->api->allocGlobMem(N * sizeof(real));
  std::vector<real> hostVector(N, 0);

  device->api->copyTo(arr, &hostVector[0], N * sizeof(real));
  device->algorithms.touchMemory(arr, N, false);
  device->api->copyFrom(&hostVector[0], arr, N * sizeof(real));

  for (auto &i : hostVector) {
    EXPECT_EQ(0, i);
  }

  device->api->freeMem(arr);
}

TEST_F(Reductions, scale) {
  const int N = 100;
  std::vector<int> hostVector(N, 1);
  int *arr = (int *)device->api->allocGlobMem(N * sizeof(int));

  device->api->copyTo(arr, &hostVector[0], N * sizeof(int));
  device->algorithms.scaleArray(arr, 5, N);
  device->api->copyFrom(&hostVector[0], arr, N * sizeof(int));

  for (auto &i : hostVector) {
    EXPECT_EQ(5, i);
  }

  device->api->freeMem(arr);
}

TEST_F(Reductions, streamBatchedData) {
  const int N = 100;
  std::vector<int> hostVector(N, 1);
  int *arr = (int *)device->api->allocGlobMem(N * sizeof(int));

  device->api->copyTo(arr, &hostVector[0], N * sizeof(int));
  device->algorithms.scaleArray(arr, 5, N);
  device->api->copyFrom(&hostVector[0], arr, N * sizeof(int));

  for (auto &i : hostVector) {
    EXPECT_EQ(5, i);
  }

  device->api->freeMem(arr);
}

TEST_F(Reductions, copy2DMemory) {
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

TEST_F(Reductions, copy2DMemoryWithSrcPitch) {

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

TEST_F(Reductions, copy2DMemoryWithDstPitch) {

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