// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "BaseTestSuite.h"
#include "device.h"

#include "gtest/gtest.h"
#include <cstddef>
#include <vector>

using namespace device;
using namespace ::testing;

namespace {
constexpr std::size_t BatchSize = 64;
constexpr std::size_t ElementSize = 48;
} // namespace

/**
 * Covers the batched transfers and the pointer arithmetic helper, which the other suites do not
 * touch. These are the operations the solver uses to move data in and out of batched buffers, so
 * a wrong stride or a skipped entry shows up as a wrong answer far away from here.
 */
class BatchTransfer : public BaseTestSuite {
  public:
  void SetUp() override {
    BaseTestSuite::SetUp();
    stream = device->api->createStream();
    src = static_cast<float*>(device->api->allocGlobMem(BatchSize * ElementSize * sizeof(float)));
    dst = static_cast<float*>(device->api->allocGlobMem(BatchSize * ElementSize * sizeof(float)));
    srcBatch = static_cast<float**>(device->api->allocUnifiedMem(BatchSize * sizeof(float*)));
    dstBatch = static_cast<float**>(device->api->allocUnifiedMem(BatchSize * sizeof(float*)));

    for (std::size_t i = 0; i < BatchSize; ++i) {
      srcBatch[i] = src + i * ElementSize;
      dstBatch[i] = dst + i * ElementSize;
    }
  }

  void TearDown() override {
    device->api->freeUnifiedMem(dstBatch);
    device->api->freeUnifiedMem(srcBatch);
    device->api->freeGlobMem(dst);
    device->api->freeGlobMem(src);
    device->api->destroyGenericStream(stream);
  }

  protected:
  void upload(float* target, const std::vector<float>& host) {
    device->api->copyToAsync(target, host.data(), host.size() * sizeof(float), stream);
    device->api->syncStreamWithHost(stream);
  }

  std::vector<float> download(const float* source) {
    std::vector<float> host(BatchSize * ElementSize, -1);
    device->api->copyFromAsync(host.data(), source, host.size() * sizeof(float), stream);
    device->api->syncStreamWithHost(stream);
    return host;
  }

  // a value that differs per batch entry and per element, so a mixed-up stride cannot pass
  static float pattern(std::size_t entry, std::size_t element) {
    return static_cast<float>(entry * ElementSize + element);
  }

  void* stream{nullptr};
  float* src{nullptr};
  float* dst{nullptr};
  float** srcBatch{nullptr};
  float** dstBatch{nullptr};
};

TEST_F(BatchTransfer, streamBatchedDataCopiesEveryEntry) {
  std::vector<float> hostSrc(BatchSize * ElementSize);
  for (std::size_t i = 0; i < BatchSize; ++i) {
    for (std::size_t j = 0; j < ElementSize; ++j) {
      hostSrc[i * ElementSize + j] = pattern(i, j);
    }
  }
  upload(src, hostSrc);
  upload(dst, std::vector<float>(BatchSize * ElementSize, 0.0F));

  device->algorithms.streamBatchedData(
      const_cast<const float**>(srcBatch), dstBatch, ElementSize, BatchSize, stream);
  device->api->syncStreamWithHost(stream);

  const auto hostDst = download(dst);
  for (std::size_t i = 0; i < BatchSize; ++i) {
    for (std::size_t j = 0; j < ElementSize; ++j) {
      ASSERT_EQ(pattern(i, j), hostDst[i * ElementSize + j])
          << "at entry " << i << ", element " << j;
    }
  }
}

TEST_F(BatchTransfer, streamBatchedDataSkipsNullEntries) {
  upload(src, std::vector<float>(BatchSize * ElementSize, 1.0F));
  upload(dst, std::vector<float>(BatchSize * ElementSize, -3.0F));

  for (std::size_t i = 0; i < BatchSize; i += 2) {
    srcBatch[i] = nullptr;
  }

  device->algorithms.streamBatchedData(
      const_cast<const float**>(srcBatch), dstBatch, ElementSize, BatchSize, stream);
  device->api->syncStreamWithHost(stream);

  const auto hostDst = download(dst);
  for (std::size_t i = 0; i < BatchSize; ++i) {
    const float expected = (i % 2 == 0) ? -3.0F : 1.0F;
    for (std::size_t j = 0; j < ElementSize; ++j) {
      ASSERT_EQ(expected, hostDst[i * ElementSize + j]) << "at entry " << i << ", element " << j;
    }
  }
}

TEST_F(BatchTransfer, accumulateBatchedDataAdds) {
  std::vector<float> hostSrc(BatchSize * ElementSize);
  for (std::size_t i = 0; i < BatchSize; ++i) {
    for (std::size_t j = 0; j < ElementSize; ++j) {
      hostSrc[i * ElementSize + j] = pattern(i, j);
    }
  }
  upload(src, hostSrc);
  upload(dst, std::vector<float>(BatchSize * ElementSize, 5.0F));

  device->algorithms.accumulateBatchedData(
      const_cast<const float**>(srcBatch), dstBatch, ElementSize, BatchSize, stream);
  device->api->syncStreamWithHost(stream);

  const auto hostDst = download(dst);
  for (std::size_t i = 0; i < BatchSize; ++i) {
    for (std::size_t j = 0; j < ElementSize; ++j) {
      ASSERT_EQ(5.0F + pattern(i, j), hostDst[i * ElementSize + j])
          << "at entry " << i << ", element " << j;
    }
  }
}

TEST_F(BatchTransfer, accumulateBatchedDataIsRepeatable) {
  upload(src, std::vector<float>(BatchSize * ElementSize, 2.0F));
  upload(dst, std::vector<float>(BatchSize * ElementSize, 0.0F));

  for (int round = 0; round < 3; ++round) {
    device->algorithms.accumulateBatchedData(
        const_cast<const float**>(srcBatch), dstBatch, ElementSize, BatchSize, stream);
  }
  device->api->syncStreamWithHost(stream);

  for (const auto value : download(dst)) {
    ASSERT_EQ(6.0F, value);
  }
}

TEST_F(BatchTransfer, incrementalAddBuildsAStridedPointerTable) {
  auto** table = static_cast<float**>(device->api->allocUnifiedMem(BatchSize * sizeof(float*)));

  device->algorithms.incrementalAdd(table, src, ElementSize, BatchSize, stream);
  device->api->syncStreamWithHost(stream);

  // the stride is given in elements, not bytes
  for (std::size_t i = 0; i < BatchSize; ++i) {
    ASSERT_EQ(src + i * ElementSize, table[i]) << "at entry " << i;
  }

  device->api->freeUnifiedMem(table);
}
