#include "device.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>
#include <random>
#include <functional>
#include <numeric>
#include <iostream>

using namespace device;
using namespace ::testing;

class Reductions : public ::testing::Test {
public:
  Reductions() {
    device.api->initialize();
    device.api->allocateStackMem();
    randomEngine.seed(randomDevice());
  }
  virtual ~Reductions() {
    device.api->finalize();
  }

protected:
  void SetUp() override {}

  DeviceInstance &device = DeviceInstance::getInstance();
  std::random_device randomDevice;
  std::mt19937 randomEngine;
};


TEST_F(Reductions, Add) {
  constexpr size_t size = 1000;
  std::vector<int> vector(size, 0);

  std::uniform_int_distribution<> distribution(-50, 50);
  for (auto& element: vector) {
    element = distribution(randomEngine);
  }

  int* devVector = reinterpret_cast<int*>(device.api->getStackMemory(sizeof(int) * size));
  device.api->copyTo(devVector, vector.data(), sizeof(int) * size);

  int expectedResult = std::accumulate(vector.begin(), vector.end(), 0, std::plus<int>());

  int testResult = device.algorithms.reduceVector(devVector, size, ReductionType::Add);
  EXPECT_EQ(expectedResult, testResult);

  device.api->popStackMemory();
}


TEST_F(Reductions, Max) {
  constexpr size_t size = 1000;
  std::vector<int> vector(size, 0);

  std::uniform_int_distribution<> distribution(-50, 50);
  for (auto& element: vector) {
    element = distribution(randomEngine);
  }

  int* devVector = reinterpret_cast<int*>(device.api->getStackMemory(sizeof(int) * size));
  device.api->copyTo(devVector, vector.data(), sizeof(int) * size);

  auto max = [](int a, int b) -> int {
    return a > b ? a : b;
  };
  int expectedResult = std::accumulate(vector.begin(), vector.end(), 0, max);

  int testResult = device.algorithms.reduceVector(devVector, size, ReductionType::Max);
  EXPECT_EQ(expectedResult, testResult);

  device.api->popStackMemory();
}


TEST_F(Reductions, MIN) {
  constexpr size_t size = 1000;
  std::vector<int> vector(size, 0);

  std::uniform_int_distribution<> distribution(-50, 50);
  for (auto& element: vector) {
    element = distribution(randomEngine);
  }

  int* devVector = reinterpret_cast<int*>(device.api->getStackMemory(sizeof(int) * size));
  device.api->copyTo(devVector, vector.data(), sizeof(int) * size);

  auto min = [](int a, int b) -> int {
    return a > b ? b : a;
  };
  int expectedResult = std::accumulate(vector.begin(), vector.end(), 0, min);

  int testResult = device.algorithms.reduceVector(devVector, size, ReductionType::Min);
  EXPECT_EQ(expectedResult, testResult);

  device.api->popStackMemory();
}