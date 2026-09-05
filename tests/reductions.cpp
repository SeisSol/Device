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

/**
 * The reductions above run over unsigned values only, where the neutral element of a maximum and
 * the smallest representable value are the same thing. They are not for signed integers, and for
 * floating point types numeric_limits<T>::min() is the smallest positive normal value, so a
 * maximum over negative data has to start below all of them to come out right.
 */
template <typename T>
class SignedReductions : public BaseTestSuite {
  protected:
  void run(ReductionType type, const std::vector<T>& host, T expected) {
    auto* devVector = static_cast<T*>(device->api->allocGlobMem(sizeof(T) * host.size()));
    device->api->copyTo(devVector, host.data(), sizeof(T) * host.size());

    auto* result = static_cast<T*>(device->api->allocPinnedMem(sizeof(T)));
    *result = T{0};

    device->algorithms.reduceVector(
        result, devVector, true, host.size(), type, device->api->getDefaultStream());
    device->api->syncDefaultStreamWithHost();

    EXPECT_EQ(expected, *result);

    device->api->freePinnedMem(result);
    device->api->freeGlobMem(devVector);
  }
};

using SignedTypes = ::testing::Types<int, long, float, double>;
TYPED_TEST_SUITE(SignedReductions, SignedTypes);

TYPED_TEST(SignedReductions, maxOverNegativeValues) {
  std::vector<TypeParam> host(100000, TypeParam{-7});
  host[host.size() / 3] = TypeParam{-2};
  this->run(ReductionType::Max, host, TypeParam{-2});
}

TYPED_TEST(SignedReductions, minOverNegativeValues) {
  std::vector<TypeParam> host(100000, TypeParam{-7});
  host[host.size() / 3] = TypeParam{-11};
  this->run(ReductionType::Min, host, TypeParam{-11});
}

TYPED_TEST(SignedReductions, addOverNegativeValues) {
  std::vector<TypeParam> host(1000, TypeParam{-3});
  this->run(ReductionType::Add, host, TypeParam{-3000});
}

TEST_F(Reductions, emptyInput) {
  auto* devVector = static_cast<float*>(device->api->allocGlobMem(sizeof(float)));
  auto* result = static_cast<float*>(device->api->allocPinnedMem(sizeof(float)));
  *result = 123.0F;

  // nothing to reduce, but the result is still initialized to the neutral element
  device->algorithms.reduceVector(
      result, devVector, true, 0, ReductionType::Add, device->api->getDefaultStream());
  device->api->syncDefaultStreamWithHost();
  EXPECT_EQ(0.0F, *result);

  device->api->freePinnedMem(result);
  device->api->freeGlobMem(devVector);
}
