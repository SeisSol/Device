// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "BaseTestSuite.h"
#include "device.h"
#include "gtest/gtest.h"
#include <vector>

using namespace device;
using namespace ::testing;

class ArrayManip : public BaseTestSuite {
  using BaseTestSuite::BaseTestSuite;
};

TEST_F(ArrayManip, fill) {
  const int N = 100;
  int *arr = (int *)device->api->allocGlobMem(N * sizeof(int));
  int scalar = 502;

  device->algorithms.fillArray(arr, scalar, N, device->api->getDefaultStream());

  std::vector<int> hostVector(N, 0);
  device->api->copyFromAsync(&hostVector[0], arr, N * sizeof(int), device->api->getDefaultStream());

  device->api->syncDefaultStreamWithHost();

  for (auto &i : hostVector) {
    EXPECT_EQ(scalar, i);
  }

  device->api->freeMem(arr);
}

TEST_F(ArrayManip, touchClean) {

  const int N = 100;
  real *arr = (real *)device->api->allocGlobMem(N * sizeof(real));
  device->algorithms.touchMemory(arr, N, true, device->api->getDefaultStream());
  std::vector<real> hostVector(N, 1);

  device->api->copyFromAsync(&hostVector[0], arr, N * sizeof(real), device->api->getDefaultStream());

  device->api->syncDefaultStreamWithHost();

  for (auto &i : hostVector) {
    EXPECT_EQ(0, i);
  }

  device->api->freeMem(arr);
}

TEST_F(ArrayManip, touchNoClean) {

  const int N = 100;
  real *arr = (real *)device->api->allocGlobMem(N * sizeof(real));
  std::vector<real> hostVector(N, 0);

  device->api->copyToAsync(arr, &hostVector[0], N * sizeof(real), device->api->getDefaultStream());
  device->algorithms.touchMemory(arr, N, false, device->api->getDefaultStream());
  device->api->copyFromAsync(&hostVector[0], arr, N * sizeof(real), device->api->getDefaultStream());

  device->api->syncDefaultStreamWithHost();

  for (auto &i : hostVector) {
    EXPECT_EQ(0, i);
  }

  device->api->freeMem(arr);
}

TEST_F(ArrayManip, scale) {
  const int N = 100;
  std::vector<int> hostVector(N, 1);
  int *arr = (int *)device->api->allocGlobMem(N * sizeof(int));

  device->api->copyToAsync(arr, &hostVector[0], N * sizeof(int), device->api->getDefaultStream());
  device->algorithms.scaleArray(arr, 5, N, device->api->getDefaultStream());
  device->api->copyFromAsync(&hostVector[0], arr, N * sizeof(int), device->api->getDefaultStream());

  device->api->syncDefaultStreamWithHost();

  for (auto &i : hostVector) {
    EXPECT_EQ(5, i);
  }

  device->api->freeMem(arr);
}

