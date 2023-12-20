#include "BaseTestSuite.h"
#include "device.h"
#include "gtest/gtest.h"
#include <functional>
#include <vector>

using namespace device;
using namespace ::testing;

class BatchManip : public BaseTestSuite {
  using BaseTestSuite::BaseTestSuite;

  template<typename T, typename F>
  void testWrapper(size_t N, bool sparse, F&& inner) {
    int *data = (int *)device->api->allocGlobMem(N * sizeof(T));
    int **batch = (int **)device->api->allocUnifiedMem(N * sizeof(T*));

    for (size_t i = 0; i < N; ++i) {
      if (!(sparse && i % 2 == 0)) {
        batch[i] = data + i;
      }
      else {
        batch[i] = nullptr;
      }
    }

    std::invoke(std::forward<F>(inner), batch, data);

    device->api->freeMem(data);
    device->api->freeMem(batch);
  }
};

TEST_F(BatchManip, fill) {
  const int N = 100;
  testWrapper(N, false, [](int** batch, int* data) {
    int scalar = 502;

    device->algorithms.fillArray(batch, scalar, N, device->api->getDefaultStream());

    std::vector<int> hostVector(N, 0);
    device->api->copyFromAsync(&hostVector[0], data, N * sizeof(int), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
        EXPECT_EQ(scalar, i);
    }
  });
}

TEST_F(BatchManip, touchClean) {
  const int N = 100;
  testWrapper(N, false, [](int** batch, int* data) {
    device->algorithms.touchMemory(batch, N, true, device->api->getDefaultStream());
    std::vector<real> hostVector(N, 1);

    device->api->copyFromAsync(&hostVector[0], data, N * sizeof(real), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
      EXPECT_EQ(0, i);
    }
  });

  testWrapper(N, true, [](int** batch, int* data) {
    device->algorithms.touchMemory(batch, N, true, device->api->getDefaultStream());
    std::vector<real> hostVector(N, 0);

    device->api->copyFromAsync(&hostVector[0], data, N * sizeof(real), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (size_t i = 0; i < hostVector.size(); ++i) {
      if (i % 2 == 0) {
        EXPECT_EQ(0, hostVector[i]);
      }
      else {
        EXPECT_EQ(1, hostVector[i]);
      }
    }
  });
}

TEST_F(BatchManip, touchNoClean) {

  const int N = 100;
  testWrapper(N, true, [](int** batch, int* data) {
    std::vector<real> hostVector(N, 1);

    device->api->copyToAsync(arr, &hostVector[0], N * sizeof(real), device->api->getDefaultStream());
    device->algorithms.touchMemory(batch, N, false, device->api->getDefaultStream());
    device->api->copyFromAsync(&hostVector[0], arr, N * sizeof(real), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
      EXPECT_EQ(0, i);
    }
  });
}

TEST_F(BatchManip, scale) {
  const int N = 100;
  testWrapper(N, true, [](int** batch, int* data) {
    std::vector<int> hostVector(N, 1);

    device->api->copyToAsync(data, &hostVector[0], N * sizeof(int), device->api->getDefaultStream());
    device->algorithms.scaleArray(batch, 5, N, device->api->getDefaultStream());
    device->api->copyFromAsync(&hostVector[0], data, N * sizeof(int), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
      EXPECT_EQ(5, i);
    }
  });
}
