// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "BaseTestSuite.h"
#include "device.h"

#include "gtest/gtest.h"
#include <algorithm>
#include <cstddef>

using namespace device;
using namespace ::testing;

class Streams : public BaseTestSuite {
  using BaseTestSuite::BaseTestSuite;
};

TEST_F(Streams, hostFunctionRunsAfterPrecedingWork) {
  // large enough that the copies are still running when the host function is enqueued
  constexpr std::size_t Count = std::size_t{1} << 24;

  auto* source = static_cast<int*>(device->api->allocPinnedMem(Count * sizeof(int)));
  auto* target = static_cast<int*>(device->api->allocPinnedMem(Count * sizeof(int)));
  auto* buffer = static_cast<int*>(device->api->allocGlobMem(Count * sizeof(int)));
  std::fill(source, source + Count, 1904);
  std::fill(target, target + Count, 0);

  void* stream = device->api->createStream();
  device->api->copyToAsync(buffer, source, Count * sizeof(int), stream);
  device->api->copyFromAsync(target, buffer, Count * sizeof(int), stream);

  int seen = 0;
  device->api->streamHostFunction(stream, [&]() { seen = target[Count - 1]; });
  device->api->syncStreamWithHost(stream);

  EXPECT_EQ(1904, seen);

  device->api->destroyGenericStream(stream);
  device->api->freeGlobMem(buffer);
  device->api->freePinnedMem(target);
  device->api->freePinnedMem(source);
}

TEST_F(Streams, streamWorkIsDoneAfterSync) {
  constexpr std::size_t Count = 1024;

  auto* source = static_cast<int*>(device->api->allocPinnedMem(Count * sizeof(int)));
  auto* buffer = static_cast<int*>(device->api->allocGlobMem(Count * sizeof(int)));
  std::fill(source, source + Count, 1904);

  void* stream = device->api->createStream();
  device->api->copyToAsync(buffer, source, Count * sizeof(int), stream);
  device->api->syncStreamWithHost(stream);

  // a bounded number of polls, so that a stream that never reports as done fails the test instead
  // of hanging it
  bool done = false;
  for (int poll = 0; poll < 1000 && !done; ++poll) {
    done = device->api->isStreamWorkDone(stream);
  }
  EXPECT_TRUE(done);

  device->api->destroyGenericStream(stream);
  device->api->freeGlobMem(buffer);
  device->api->freePinnedMem(source);
}
