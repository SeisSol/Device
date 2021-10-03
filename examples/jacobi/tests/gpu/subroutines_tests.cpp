#include "datatypes.hpp"
#include <device.h>
#include <gmock/gmock.h>
#include <gpu/kernels/subroutinesGPU.h>
#include <gtest/gtest.h>
#include <vector>

using ::testing::ElementsAreArray;

TEST(Subroutines, MultMatrixVec) {
  auto *api = ::device::DeviceInstance::getInstance().api;
  auto defaultStream = api->getDefaultStream();
  const int size = 3;

  WorkSpaceT space{};
  MatrixInfoT matrixInfoT{space, size, size};
  GpuMatrixDataT matrix{matrixInfoT};

  matrix.data = (real *)api->allocUnifiedMem(size * size * sizeof(real));
  matrix.indices = (int *)api->allocUnifiedMem(size * size * sizeof(int));

  for (int i = 0; i < size * size; ++i)
  {
    const int linIndex = (i / size) + (i % size) * size;
    matrix.data[linIndex] =  i + 1 ;
    matrix.indices[i] = i % size;
  }
  real v[] = {1, 1, 1};
  real res[] = {0, 0, 0};
  real ref[] = {6, 15, 24};

  auto *devV = (real *)api->allocGlobMem(size * sizeof(real));
  auto *devRes = (real *)api->allocGlobMem(size * sizeof(real));

  api->copyTo(devV, v, size * sizeof(real));
  api->copyTo(devRes, res, size * sizeof(real));

  launch_multMatVec(matrix, devV, devRes, defaultStream);

  //we dont need a sync here; all API queues are in order and this memory copy is synchronous
  api->copyFrom(res, devRes, size * sizeof(real));

  api->freeMem(matrix.data);
  api->freeMem(matrix.indices);
  api->freeMem(devV);
  api->freeMem(devRes);

  ASSERT_THAT(res, ElementsAreArray(ref));
}

TEST(Subroutines, VectorManips) {
  auto *api = ::device::DeviceInstance::getInstance().api;
  auto defaultStream = api->getDefaultStream();

  const int size = 3;
  RangeT rng {0, size};

  auto *devA = (real *)api->allocUnifiedMem(size * sizeof(real));
  auto *devB = (real *)api->allocUnifiedMem(size * sizeof(real));
  auto *devRes = (real *)api->allocUnifiedMem(size * sizeof(real));
  auto ref = std::vector<real>(size);
  auto res = std::vector<real>(size);

  for (int i = 0; i < size; ++i) {
    devA[i] = i;
    devB[i] = i;
    devRes[i] = 0;
    ref[i] = i + i;
  }

  launch_manipVectors(rng, devA, devB, devRes, VectorManipOps::Addition, defaultStream);
  api->copyFrom(&res[0], devRes, size * sizeof(real));
  ASSERT_THAT(res, ElementsAreArray(ref));

  for (int i = 0; i < size; ++i)
    ref[i] = i * i;

  launch_manipVectors(rng, devA, devB, devRes, VectorManipOps::Multiply, defaultStream);
  api->copyFrom(&res[0], devRes, size * sizeof(real));
  ASSERT_THAT(res, ElementsAreArray(ref));

  for (int i = 0; i < size; ++i)
    ref[i] = 0;

  launch_manipVectors(rng, devA, devB, devRes, VectorManipOps::Subtraction, defaultStream);
  api->copyFrom(&res[0], devRes, size * sizeof(real));
  ASSERT_THAT(res, ElementsAreArray(ref));
}