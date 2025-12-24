// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "BaseTestSuite.h"
#include "device.h"

#include "gtest/gtest.h"
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using namespace device;
using namespace ::testing;

class Reductions : public BaseTestSuite {
  using BaseTestSuite::BaseTestSuite;
};

TEST_F(Reductions, Add) {
  constexpr size_t size = 10010000;
  std::vector<unsigned> vector(size, 0);

  std::uniform_int_distribution<> distribution(10, 50);
  for (auto& element : vector) {
    element = distribution(randomEngine);
  }

  auto* devVector = reinterpret_cast<unsigned*>(device->api->allocGlobMem(sizeof(unsigned) * size));
  device->api->copyTo(devVector, vector.data(), sizeof(unsigned) * size);

  auto expectedResult = std::accumulate(vector.begin(), vector.end(), 0, std::plus<unsigned>());

  unsigned* testResult = reinterpret_cast<unsigned*>(device->api->allocPinnedMem(sizeof(unsigned)));

  device->algorithms.reduceVector(
      testResult, devVector, true, size, ReductionType::Add, device->api->getDefaultStream());
  device->api->syncDefaultStreamWithHost();
  EXPECT_EQ(expectedResult, *testResult);
  device->api->freePinnedMem(testResult);
  device->api->freeGlobMem(devVector);
}

TEST_F(Reductions, Max) {
  constexpr size_t size = 20010000;
  std::vector<unsigned> vector(size, 0);

  auto* devVector = reinterpret_cast<unsigned*>(device->api->allocGlobMem(sizeof(unsigned) * size));
  device->api->copyTo(devVector, vector.data(), sizeof(unsigned) * size);

  std::uniform_int_distribution<> distribution(10, 100);
  for (auto& element : vector) {
    element = distribution(randomEngine);
  }

  device->api->copyTo(devVector, vector.data(), sizeof(unsigned) * size);

  auto max = [](unsigned a, unsigned b) -> unsigned { return a > b ? a : b; };
  auto initValue = std::numeric_limits<unsigned>::min();
  auto expectedResult = std::accumulate(vector.begin(), vector.end(), initValue, max);

  unsigned* testResult = reinterpret_cast<unsigned*>(device->api->allocPinnedMem(sizeof(unsigned)));

  device->algorithms.reduceVector(
      testResult, devVector, true, size, ReductionType::Max, device->api->getDefaultStream());
  device->api->syncDefaultStreamWithHost();
  EXPECT_EQ(expectedResult, *testResult);
  device->api->freePinnedMem(testResult);
  device->api->freeGlobMem(devVector);
}

TEST_F(Reductions, Min) {
  constexpr size_t size = 30020000;
  std::vector<unsigned> vector(size, 0);

  std::uniform_int_distribution<> distribution(10, 100);
  for (auto& element : vector) {
    element = distribution(randomEngine);
  }

  auto* devVector = reinterpret_cast<unsigned*>(device->api->allocGlobMem(sizeof(unsigned) * size));
  device->api->copyTo(devVector, vector.data(), sizeof(unsigned) * size);

  auto min = [](unsigned a, unsigned b) -> unsigned { return a > b ? b : a; };
  auto initValue = std::numeric_limits<unsigned>::max();
  auto expectedResult = std::accumulate(vector.begin(), vector.end(), initValue, min);

  unsigned* testResult = reinterpret_cast<unsigned*>(device->api->allocPinnedMem(sizeof(unsigned)));

  device->algorithms.reduceVector(
      testResult, devVector, true, size, ReductionType::Min, device->api->getDefaultStream());
  device->api->syncDefaultStreamWithHost();
  EXPECT_EQ(expectedResult, *testResult);
  device->api->freePinnedMem(testResult);
  device->api->freeGlobMem(devVector);
}
