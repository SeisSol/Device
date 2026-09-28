// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "BaseTestSuite.h"
#include "device.h"

#include "gtest/gtest.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace device;
using namespace ::testing;

// the behavior that the host backend (DEVICE_BACKEND=none) guarantees on top of the common API
class Host : public BaseTestSuite {
  using BaseTestSuite::BaseTestSuite;
};

namespace {
size_t leakedBytes(AbstractAPI& api) {
  const std::string report = api.getMemLeaksReport();
  const std::string prefix = "bytes: ";
  return std::stoul(report.substr(report.find(prefix) + prefix.size()));
}
} // namespace

TEST_F(Host, properties) {
  auto& api = device->api();
  EXPECT_EQ(api.getApiName(), "Host");
  EXPECT_EQ(api.getNumDevices(), 1);
  EXPECT_EQ(api.getDeviceId(), 0);
  EXPECT_FALSE(api.isCapableOfGraphCapturing());
  EXPECT_TRUE(api.isUnifiedMemoryDefault());
  EXPECT_GT(api.getMaxAvailableMem(), 0);
}

TEST_F(Host, allocationsAreAligned) {
  auto& api = device->api();
  const auto alignment = api.getGlobMemAlignment();
  for (const size_t size : {1, 7, 128, 1000}) {
    void* glob = api.allocGlobMem(size);
    void* unified = api.allocUnifiedMem(size);
    void* pinned = api.allocPinnedMem(size);
    for (void* ptr : {glob, unified, pinned}) {
      ASSERT_NE(ptr, nullptr);
      EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignment, 0);
    }
    api.freeGlobMem(glob);
    api.freeUnifiedMem(unified);
    api.freePinnedMem(pinned);
  }
  EXPECT_EQ(api.allocGlobMem(0), nullptr);
  api.freeGlobMem(nullptr);
}

TEST_F(Host, bookkeeping) {
  auto& api = device->api();
  const auto occupied = api.getCurrentlyOccupiedMem();
  const auto occupiedUnified = api.getCurrentlyOccupiedUnifiedMem();
  const auto leaked = leakedBytes(api);

  void* glob = api.allocGlobMem(100);
  void* unified = api.allocUnifiedMem(50);
  EXPECT_EQ(api.getCurrentlyOccupiedMem(), occupied + 150);
  EXPECT_EQ(api.getCurrentlyOccupiedUnifiedMem(), occupiedUnified + 50);
  EXPECT_EQ(leakedBytes(api), leaked + 150);

  api.freeGlobMem(glob);
  api.freeUnifiedMem(unified);
  EXPECT_EQ(leakedBytes(api), leaked);
}

TEST_F(Host, hostFunctionsRunInOrder) {
  auto& api = device->api();
  void* stream = api.createStream();
  std::vector<int> order;
  api.streamHostFunction(stream, [&] { order.push_back(1); });
  api.streamHostFunction(stream, [&] { order.push_back(2); });
  api.syncStreamWithHost(stream);
  EXPECT_EQ(order, (std::vector<int>{1, 2}));
  EXPECT_TRUE(api.isStreamWorkDone(stream));
  api.destroyGenericStream(stream);
}

TEST_F(Host, eventsMeasureTime) {
  auto& api = device->api();
  void* stream = api.getDefaultStream();
  void* start = api.createEvent(true);
  void* end = api.createEvent(true);
  api.recordEventOnStream(start, stream);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  api.recordEventOnStream(end, stream);
  api.syncEventWithHost(end);
  EXPECT_TRUE(api.isEventCompleted(end));
  EXPECT_GE(api.timespanEvents(start, end), 0.019);
  api.destroyEvent(start);
  api.destroyEvent(end);
}

TEST_F(Host, waitMemory) {
  auto& api = device->api();
  auto* flag = static_cast<uint32_t*>(
      api.allocPinnedMem(sizeof(uint32_t), false, Destination::CurrentDevice));

  // already reached: returns right away
  __atomic_store_n(flag, 5U, __ATOMIC_RELEASE);
  api.streamWaitMemory(api.getDefaultStream(), flag, 3);

  // reached later, by another thread
  __atomic_store_n(flag, 0U, __ATOMIC_RELEASE);
  std::thread writer([flag] {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    __atomic_store_n(flag, 2U, __ATOMIC_RELEASE);
  });
  api.streamWaitMemory(api.getDefaultStream(), flag, 2);
  EXPECT_EQ(__atomic_load_n(flag, __ATOMIC_ACQUIRE), 2U);
  writer.join();

  api.freePinnedMem(flag);
}

TEST_F(Host, graphCapturingIsUnavailable) {
  auto& api = device->api();
  std::vector<void*> streams{api.getDefaultStream()};
  const auto handle = api.streamBeginCapture(streams);
  EXPECT_FALSE(handle.isInitialized());
  api.streamEndCapture(handle);
}

TEST_F(Host, reductions) {
  auto& api = device->api();
  const std::vector<float> values{-3.0F, -1.5F, -7.25F};
  const auto bytes = values.size() * sizeof(float);
  auto* buffer = static_cast<float*>(api.allocGlobMem(bytes));
  api.copyTo(buffer, values.data(), bytes);
  auto* result = static_cast<float*>(api.allocPinnedMem(sizeof(float)));
  void* stream = api.getDefaultStream();

  // the neutral element of the maximum is the lowest value, not the smallest positive one
  device->algorithms().reduceVector(
      result, buffer, true, values.size(), ReductionType::Max, stream);
  EXPECT_EQ(*result, -1.5F);

  device->algorithms().reduceVector(
      result, buffer, true, values.size(), ReductionType::Min, stream);
  EXPECT_EQ(*result, -7.25F);

  // without overriding, the reduction continues from the value in the result
  *result = 10.0F;
  device->algorithms().reduceVector(
      result, buffer, false, values.size(), ReductionType::Add, stream);
  EXPECT_EQ(*result, -1.75F);

  api.freePinnedMem(result);
  api.freeGlobMem(buffer);
}

TEST_F(Host, pitchedCopies) {
  auto& api = device->api();
  constexpr size_t Width = 3;
  constexpr size_t Height = 4;
  constexpr size_t SrcPitch = 5;
  constexpr size_t DevPitch = 7;
  constexpr char Padding = '#';

  std::vector<char> src(SrcPitch * Height, Padding);
  for (size_t row = 0; row < Height; ++row) {
    for (size_t column = 0; column < Width; ++column) {
      src[row * SrcPitch + column] = static_cast<char>('a' + row * Width + column);
    }
  }

  auto* dev = static_cast<char*>(api.allocGlobMem(DevPitch * Height));
  std::memset(dev, Padding, DevPitch * Height);
  api.copy2dArrayTo(dev, DevPitch, src.data(), SrcPitch, Width, Height);

  std::vector<char> dst(SrcPitch * Height, Padding);
  api.copy2dArrayFrom(dst.data(), SrcPitch, dev, DevPitch, Width, Height);
  EXPECT_EQ(dst, src);

  for (size_t row = 0; row < Height; ++row) {
    for (size_t column = Width; column < DevPitch; ++column) {
      EXPECT_EQ(dev[row * DevPitch + column], Padding);
    }
  }

  api.freeGlobMem(dev);
}
