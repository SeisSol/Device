#include "BaseTestSuite.h"
#include "device.h"
#include "gtest/gtest.h"
#include <vector>

using namespace device;
using namespace ::testing;

class ArrayManips : public BaseTestSuite {
  using BaseTestSuite::BaseTestSuite;
};

TEST_F(ArrayManips, fill) {

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

TEST_F(ArrayManips, touchClean) {

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

TEST_F(ArrayManips, touchNoClean) {

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

TEST_F(ArrayManips, scale) {
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
