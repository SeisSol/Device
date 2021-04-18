#include "device.h"
#include "gtest/gtest.h"
#include <functional>
#include <numeric>
#include <random>
#include <vector>
#include "BaseTestSuite.h"

using namespace device;
using namespace ::testing;

class Reductions : public BaseTestSuite {
  using BaseTestSuite::BaseTestSuite;
};


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
