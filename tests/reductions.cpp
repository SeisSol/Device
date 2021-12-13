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
  std::vector<unsigned> vector(size, 0);

  std::uniform_int_distribution<> distribution(10, 50);
  for (auto &element : vector) {
    element = distribution(randomEngine);
  }

  auto devVector = reinterpret_cast<unsigned *>(device->api->getStackMemory(sizeof(unsigned) * size));
  device->api->copyTo(devVector, vector.data(), sizeof(unsigned) * size);

  auto expectedResult = std::accumulate(vector.begin(), vector.end(), 0, std::plus<unsigned>());

  auto testResult = device->algorithms.reduceVector(devVector, size, ReductionType::Add, device->api->getDefaultStream());
  device->api->popStackMemory();
  EXPECT_EQ(expectedResult, testResult);
}

TEST_F(Reductions, Max) {
  constexpr size_t size = 1000;
  std::vector<unsigned> vector(size, 0);

  unsigned *devVector = reinterpret_cast<unsigned *>(device->api->getStackMemory(sizeof(unsigned) * size));
  device->api->copyTo(devVector, vector.data(), sizeof(unsigned) * size);

  std::uniform_int_distribution<> distribution(10, 100);
  for (auto &element : vector) {
    element = distribution(randomEngine);
  }

  device->api->copyTo(devVector, vector.data(), sizeof(unsigned) * size);

  auto max = [](unsigned a, unsigned b) -> unsigned { return a > b ? a : b; };
  auto expectedResult = std::accumulate(vector.begin(), vector.end(), 0, max);

  auto testResult = device->algorithms.reduceVector(devVector, size, ReductionType::Max, device->api->getDefaultStream());
  device->api->popStackMemory();
  EXPECT_EQ(expectedResult, testResult);
}

TEST_F(Reductions, Min) {
  constexpr size_t size = 1000;
  std::vector<unsigned> vector(size, 0);

  std::uniform_int_distribution<> distribution(10, 100);
  for (auto &element : vector) {
    element = distribution(randomEngine);
  }

  unsigned *devVector = reinterpret_cast<unsigned *>(device->api->getStackMemory(sizeof(unsigned) * size));
  device->api->copyTo(devVector, vector.data(), sizeof(unsigned) * size);

  auto min = [](unsigned a, unsigned b) -> unsigned { return a > b ? b : a; };
  auto expectedResult = std::accumulate(vector.begin(), vector.end(), 0, min);

  auto testResult = device->algorithms.reduceVector(devVector, size, ReductionType::Min, device->api->getDefaultStream());
  device->api->popStackMemory();
  EXPECT_EQ(expectedResult, testResult);
}
