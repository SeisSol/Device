// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "BaseTestSuite.h"
#include "device.h"

#include "gtest/gtest.h"
#include <atomic>
#include <cstddef>
#include <vector>

using namespace device;
using namespace ::testing;

namespace {
constexpr std::size_t ArraySize = 1 << 14;

// Enough queued work that the host is certain to run ahead of the producing stream. Without it,
// an ordering test passes whenever the producer happens to finish first, so a missing dependency
// shows up as an occasional failure rather than as a verdict.
constexpr int EnqueueDepth = 64;
} // namespace

class Streams : public BaseTestSuite {
  public:
  void SetUp() override {
    BaseTestSuite::SetUp();
    devArray = static_cast<float*>(device->api->allocGlobMem(ArraySize * sizeof(float)));
    streamA = device->api->createStream();
    streamB = device->api->createStream();
  }

  void TearDown() override {
    device->api->destroyGenericStream(streamB);
    device->api->destroyGenericStream(streamA);
    device->api->freeGlobMem(devArray);
  }

  protected:
  std::vector<float> download(void* stream) {
    std::vector<float> host(ArraySize, -1);
    device->api->copyFromAsync(host.data(), devArray, ArraySize * sizeof(float), stream);
    device->api->syncStreamWithHost(stream);
    return host;
  }

  float* devArray{nullptr};
  void* streamA{nullptr};
  void* streamB{nullptr};
};

TEST_F(Streams, anEventOrdersTwoStreams) {
  auto* event = device->api->createEvent();

  for (int i = 0; i < EnqueueDepth; ++i) {
    device->algorithms.fillArray(devArray, 2.0F, ArraySize, streamA);
  }
  device->api->recordEventOnStream(event, streamA);

  device->api->syncStreamWithEvent(streamB, event);
  device->algorithms.scaleArray(devArray, 7.0F, ArraySize, streamB);

  device->api->syncStreamWithHost(streamB);

  // a 7 here means the scale read the array before the fills wrote it
  for (const auto value : download(streamB)) {
    ASSERT_EQ(14.0F, value);
  }

  device->api->destroyEvent(event);
}

TEST_F(Streams, anEventCanBeRecordedAgain) {
  // the stream runtime hands the same event out repeatedly, so re-recording one that has already
  // been waited upon has to keep working
  auto* event = device->api->createEvent();

  for (int round = 1; round <= 4; ++round) {
    for (int i = 0; i < EnqueueDepth; ++i) {
      device->algorithms.fillArray(devArray, static_cast<float>(round), ArraySize, streamA);
    }
    device->api->recordEventOnStream(event, streamA);
    device->api->syncStreamWithEvent(streamB, event);
    device->algorithms.scaleArray(devArray, 2.0F, ArraySize, streamB);
    device->api->syncStreamWithHost(streamB);

    // twice the previous round's value means the scale overtook this round's fills
    for (const auto value : download(streamB)) {
      ASSERT_EQ(2.0F * static_cast<float>(round), value)
          << "in round " << round << " (twice the previous round's value would be "
          << 4.0F * static_cast<float>(round - 1) << ")";
    }
  }

  device->api->destroyEvent(event);
}

TEST_F(Streams, anEventIsCompleteOnceItsStreamIs) {
  auto* event = device->api->createEvent();

  device->algorithms.fillArray(devArray, 1.0F, ArraySize, streamA);
  device->api->recordEventOnStream(event, streamA);
  device->api->syncStreamWithHost(streamA);

  // only the direction after synchronizing is deterministic; whether the event is already
  // complete beforehand depends on timing
  EXPECT_TRUE(device->api->isEventCompleted(event));
  EXPECT_TRUE(device->api->isStreamWorkDone(streamA));

  device->api->destroyEvent(event);
}

TEST_F(Streams, aHostFunctionRunsInStreamOrder) {
  std::vector<float> staging(ArraySize, -1);
  std::atomic<bool> sawFilledData{false};
  std::atomic<bool> ran{false};

  device->algorithms.fillArray(devArray, 9.0F, ArraySize, streamA);
  device->api->copyFromAsync(staging.data(), devArray, ArraySize * sizeof(float), streamA);
  device->api->streamHostFunction(streamA, [&]() {
    ran = true;
    sawFilledData = (staging.front() == 9.0F) && (staging.back() == 9.0F);
  });

  device->api->syncStreamWithHost(streamA);

  EXPECT_TRUE(ran.load());
  // the callback must not observe the staging buffer before the copy that precedes it
  EXPECT_TRUE(sawFilledData.load());
}

TEST_F(Streams, asyncAllocationsLiveOnTheStream) {
  auto* scratch =
      static_cast<float*>(device->api->allocMemAsync(ArraySize * sizeof(float), streamA));
  ASSERT_NE(nullptr, scratch);

  device->algorithms.fillArray(scratch, 4.0F, ArraySize, streamA);

  std::vector<float> host(ArraySize, -1);
  device->api->copyFromAsync(host.data(), scratch, ArraySize * sizeof(float), streamA);

  // the free is ordered behind the copy on the same stream
  device->api->freeMemAsync(scratch, streamA);
  device->api->syncStreamWithHost(streamA);

  for (const auto value : host) {
    ASSERT_EQ(4.0F, value);
  }
}

TEST_F(Streams, aDestroyedStreamIsForgotten) {
  // A device-wide synchronization walks every stream the backend knows about. A stream that was
  // destroyed therefore has to be off that list, or the walk runs into freed memory - long after
  // the code that destroyed it, which is what makes this kind of fault hard to place.
  auto* scratch = device->api->createStream();
  device->algorithms.fillArray(devArray, 1.0F, ArraySize, scratch);
  device->api->syncStreamWithHost(scratch);
  device->api->destroyGenericStream(scratch);

  device->api->syncDevice();

  for (const auto value : download(streamA)) {
    ASSERT_EQ(1.0F, value);
  }
}

TEST_F(Streams, workOnSeparateStreamsStaysSeparate) {
  auto* other = static_cast<float*>(device->api->allocGlobMem(ArraySize * sizeof(float)));

  device->algorithms.fillArray(devArray, 1.0F, ArraySize, streamA);
  device->algorithms.fillArray(other, 2.0F, ArraySize, streamB);

  device->api->syncStreamWithHost(streamA);
  device->api->syncStreamWithHost(streamB);

  std::vector<float> hostOther(ArraySize, -1);
  device->api->copyFromAsync(hostOther.data(), other, ArraySize * sizeof(float), streamB);
  device->api->syncStreamWithHost(streamB);

  for (const auto value : download(streamA)) {
    ASSERT_EQ(1.0F, value);
  }
  for (const auto value : hostOther) {
    ASSERT_EQ(2.0F, value);
  }

  device->api->freeGlobMem(other);
}

TEST_F(Streams, streamsCanBeGivenAPriority) {
  // 0 is the lowest priority the device offers, 1 the highest, and the default is whatever the
  // runtime picks; all three have to give a stream that works
  for (const double priority : {0.0, 0.5, 1.0}) {
    auto* stream = device->api->createStream(priority);
    ASSERT_NE(nullptr, stream) << "at priority " << priority;

    device->algorithms.fillArray(devArray, 6.0F, ArraySize, stream);
    device->api->syncStreamWithHost(stream);

    for (const auto value : download(stream)) {
      ASSERT_EQ(6.0F, value) << "at priority " << priority;
    }

    device->api->destroyGenericStream(stream);
  }
}
