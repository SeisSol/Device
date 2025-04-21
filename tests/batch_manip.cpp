// SPDX-FileCopyrightText: 2023-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "BaseTestSuite.h"
#include "device.h"
#include "gtest/gtest.h"
#include <functional>
#include <vector>

using namespace device;
using namespace ::testing;

class BatchManip : public BaseTestSuite {
  using BaseTestSuite::BaseTestSuite;

public:
  template<typename T, typename F>
  void testWrapper(size_t N, size_t M, bool sparse, F&& inner) {
    T *data = (T *)device->api->allocGlobMem(N * M * sizeof(T));
    T **batch = (T **)device->api->allocUnifiedMem(N * sizeof(T*));

    for (size_t i = 0; i < N; ++i) {
      if (!(sparse && i % 2 == 0)) {
        batch[i] = data + i * M;
      }
      else {
        batch[i] = nullptr;
      }
    }

    std::forward<F>(inner)(batch, data);

    device->api->freeMem(data);
    device->api->freeMem(batch);
  }
};


TEST_F(BatchManip, fill32) {
  const int N = 100;
  const int M = 120;
  testWrapper<float>(N, M, false, [&](float** batch, float* data) {
    float scalar = 502;

    device->algorithms.setToValue(batch, scalar, M, N, device->api->getDefaultStream());

    std::vector<float> hostVector(N * M, 0);
    device->api->copyFromAsync(&hostVector[0], data, N * M * sizeof(float), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
        EXPECT_EQ(scalar, i);
    }
  });
}

TEST_F(BatchManip, touchClean32) {
  const int N = 100;
  const int M = 120;
  testWrapper<float>(N, M, false, [&](float** batch, float* data) {
    device->algorithms.touchBatchedMemory(batch, M, N, true, device->api->getDefaultStream());
    std::vector<float> hostVector(N * M, 1);

    device->api->copyFromAsync(&hostVector[0], data, M * N * sizeof(float), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
      EXPECT_EQ(0, i);
    }
  });

  testWrapper<float>(N, M, true, [&](float** batch, float* data) {
    std::vector<float> hostVector(N * M, 1);

    device->api->copyToAsync(data, &hostVector[0], N * M * sizeof(float), device->api->getDefaultStream());
    device->algorithms.touchBatchedMemory(batch, M, N, true, device->api->getDefaultStream());
    device->api->copyFromAsync(&hostVector[0], data, M * N * sizeof(float), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < M; ++j) {
        if (i % 2 == 0) {
          EXPECT_EQ(1, hostVector[i * M + j]);
        }
        else {
          EXPECT_EQ(0, hostVector[i * M + j]);
        }
      }
    }
  });
}

TEST_F(BatchManip, touchNoClean32) {
  const int N = 100;
  const int M = 120;
  testWrapper<float>(N, M, false, [&](float** batch, float* data) {
    std::vector<float> hostVector(N * M, 1);

    device->api->copyToAsync(data, &hostVector[0], N * M * sizeof(float), device->api->getDefaultStream());
    device->algorithms.touchBatchedMemory(batch, M, N, false, device->api->getDefaultStream());
    device->api->copyFromAsync(&hostVector[0], data, N * M * sizeof(float), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
      EXPECT_EQ(1, i);
    }
  });
}

TEST_F(BatchManip, scatterToUniform32) {
  const int N = 100;
  const int M = 120;

  float *data2 = (float *)device->api->allocGlobMem(N * M * sizeof(float));
  testWrapper<float>(N, M, false, [&](float** batch, float* data) {
    std::vector<float> hostVector(N * M, 1);

    device->api->copyToAsync(data, &hostVector[0], N * M * sizeof(float), device->api->getDefaultStream());
    device->algorithms.copyScatterToUniform(const_cast<const float**>(batch), data2, M, M, N, device->api->getDefaultStream());
    device->api->copyFromAsync(&hostVector[0], data2, N * M * sizeof(float), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
      EXPECT_EQ(1, i);
    }
  });
  device->api->freeMem(data2);
}

TEST_F(BatchManip, uniformToScatter32) {
  const int N = 10000;
  const int M = 12000;

  float *data2 = (float *)device->api->allocGlobMem(N * M * sizeof(float));
  testWrapper<float>(N, M, false, [&](float** batch, float* data) {
    std::vector<float> hostVector(N * M, 1);

    device->api->copyToAsync(data2, &hostVector[0], N * M * sizeof(float), device->api->getDefaultStream());
    device->algorithms.copyUniformToScatter(data2, batch, M, M, N, device->api->getDefaultStream());
    device->api->copyFromAsync(&hostVector[0], data, N * M * sizeof(float), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
      EXPECT_EQ(1, i);
    }
  });
  device->api->freeMem(data2);
}





TEST_F(BatchManip, fill64) {
  const int N = 100;
  const int M = 120;
  testWrapper<double>(N, M, false, [&](double** batch, double* data) {
    double scalar = 502;

    device->algorithms.setToValue(batch, scalar, M, N, device->api->getDefaultStream());

    std::vector<double> hostVector(N * M, 0);
    device->api->copyFromAsync(&hostVector[0], data, N * M * sizeof(double), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
        EXPECT_EQ(scalar, i);
    }
  });
}

TEST_F(BatchManip, touchClean64) {
  const int N = 100;
  const int M = 120;
  testWrapper<double>(N, M, false, [&](double** batch, double* data) {
    device->algorithms.touchBatchedMemory(batch, M, N, true, device->api->getDefaultStream());
    std::vector<double> hostVector(N * M, 1);

    device->api->copyFromAsync(&hostVector[0], data, M * N * sizeof(double), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
      EXPECT_EQ(0, i);
    }
  });

  testWrapper<double>(N, M, true, [&](double** batch, double* data) {
    std::vector<double> hostVector(N * M, 1);

    device->api->copyToAsync(data, &hostVector[0], N * M * sizeof(double), device->api->getDefaultStream());
    device->algorithms.touchBatchedMemory(batch, M, N, true, device->api->getDefaultStream());
    device->api->copyFromAsync(&hostVector[0], data, M * N * sizeof(double), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < M; ++j) {
        if (i % 2 == 0) {
          EXPECT_EQ(1, hostVector[i * M + j]);
        }
        else {
          EXPECT_EQ(0, hostVector[i * M + j]);
        }
      }
    }
  });
}

TEST_F(BatchManip, touchNoClean64) {
  const int N = 100;
  const int M = 120;
  testWrapper<double>(N, M, false, [&](double** batch, double* data) {
    std::vector<double> hostVector(N * M, 1);

    device->api->copyToAsync(data, &hostVector[0], N * M * sizeof(double), device->api->getDefaultStream());
    device->algorithms.touchBatchedMemory(batch, M, N, false, device->api->getDefaultStream());
    device->api->copyFromAsync(&hostVector[0], data, N * M * sizeof(double), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
      EXPECT_EQ(1, i);
    }
  });
}

TEST_F(BatchManip, scatterToUniform64) {
  const int N = 100;
  const int M = 120;

  double *data2 = (double *)device->api->allocGlobMem(N * M * sizeof(double));
  testWrapper<double>(N, M, false, [&](double** batch, double* data) {
    std::vector<double> hostVector(N * M, 1);

    device->api->copyToAsync(data, &hostVector[0], N * M * sizeof(double), device->api->getDefaultStream());
    device->algorithms.copyScatterToUniform(const_cast<const double**>(batch), data2, M, M, N, device->api->getDefaultStream());
    device->api->copyFromAsync(&hostVector[0], data2, N * M * sizeof(double), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
      EXPECT_EQ(1, i);
    }
  });
  device->api->freeMem(data2);
}

TEST_F(BatchManip, uniformToScatter64) {
  const int N = 100;
  const int M = 120;

  double *data2 = (double *)device->api->allocGlobMem(N * M * sizeof(double));
  testWrapper<double>(N, M, false, [&](double** batch, double* data) {
    std::vector<double> hostVector(N * M, 1);

    device->api->copyToAsync(data2, &hostVector[0], N * M * sizeof(double), device->api->getDefaultStream());
    device->algorithms.copyUniformToScatter(data2, batch, M, M, N, device->api->getDefaultStream());
    device->api->copyFromAsync(&hostVector[0], data, N * M * sizeof(double), device->api->getDefaultStream());

    device->api->syncDefaultStreamWithHost();

    for (auto &i : hostVector) {
      EXPECT_EQ(1, i);
    }
  });
  device->api->freeMem(data2);
}




