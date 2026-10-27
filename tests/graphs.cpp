// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "BaseTestSuite.h"
#include "device.h"

#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

using namespace device;
using namespace ::testing;

namespace {
constexpr std::size_t ArraySize = 1 << 14;
constexpr std::size_t BranchCount = 4;
constexpr std::size_t ChunkSize = ArraySize / BranchCount;
} // namespace

/**
 * The tests below check what a graph guarantees, not how it is built: every one of them states a
 * dependency structure and then asserts a result that only comes out right if that structure was
 * honoured. Operations are picked so that swapping two of them changes the answer - a fill after a
 * scale does not give the same array as a scale after a fill - because operations that commute
 * would pass no matter how the edges came out.
 */
class Graphs : public BaseTestSuite {
  public:
  void SetUp() override {
    BaseTestSuite::SetUp();
    devArray = static_cast<float*>(device->api().allocGlobMem(ArraySize * sizeof(float)));
    mainStream = device->api().createStream();
    for (auto& stream : branchStreams) {
      stream = device->api().createStream();
    }
  }

  void TearDown() override {
    for (auto* stream : branchStreams) {
      device->api().destroyGenericStream(stream);
    }
    device->api().destroyGenericStream(mainStream);
    device->api().freeGlobMem(devArray);
  }

  protected:
  std::vector<float> download() {
    std::vector<float> host(ArraySize, -1);
    device->api().copyFromAsync(host.data(), devArray, ArraySize * sizeof(float), mainStream);
    device->api().syncStreamWithHost(mainStream);
    return host;
  }

  void fill(float value) {
    device->algorithms().fillArray(devArray, value, ArraySize, mainStream);
    device->api().syncStreamWithHost(mainStream);
  }

  static void expectChunk(const std::vector<float>& host, std::size_t chunk, float value) {
    for (std::size_t i = chunk * ChunkSize; i < (chunk + 1) * ChunkSize; ++i) {
      ASSERT_EQ(value, host[i]) << "at index " << i << " of chunk " << chunk;
    }
  }

  bool graphCapturingUnavailable() { return !device->api().isCapableOfGraphCapturing(); }

  bool graphNodesUnavailable() { return !device->api().isCapableOfGraphNodes(); }

  float* devArray{nullptr};
  void* mainStream{nullptr};
  std::array<void*, BranchCount> branchStreams{};
};

TEST_F(Graphs, captureReplaysTheRecordedSequence) {
  if (graphCapturingUnavailable()) {
    GTEST_SKIP() << "the backend does not support graph capturing";
  }

  fill(0);

  std::vector<void*> streams{mainStream};
  auto graph = device->api().streamBeginCapture(streams);
  device->algorithms().fillArray(devArray, 1.0F, ArraySize, mainStream);
  device->algorithms().scaleArray(devArray, 2.0F, ArraySize, mainStream);
  device->api().streamEndCapture(graph);

  ASSERT_TRUE(graph.isInitialized());

  // nothing has run yet: capturing records, it does not execute
  for (const auto value : download()) {
    ASSERT_EQ(0.0F, value);
  }

  device->api().launchGraph(graph, mainStream);
  device->api().syncStreamWithHost(mainStream);
  for (const auto value : download()) {
    ASSERT_EQ(2.0F, value);
  }

  // the fill is part of the graph, so a second replay lands on the same value rather than doubling
  device->api().launchGraph(graph, mainStream);
  device->api().syncStreamWithHost(mainStream);
  for (const auto value : download()) {
    ASSERT_EQ(2.0F, value);
  }
}

TEST_F(Graphs, captureRecordsCrossStreamEvents) {
  if (graphCapturingUnavailable()) {
    GTEST_SKIP() << "the backend does not support graph capturing";
  }

  fill(0);

  // the fork/join shape that the stream path uses: an event hands work from the recorded stream
  // to a side stream and back. Inside a capture these become graph edges rather than real waits.
  auto* forkEvent = device->api().createEvent();
  auto* joinEvent = device->api().createEvent();

  std::vector<void*> streams{mainStream, branchStreams[0]};
  auto graph = device->api().streamBeginCapture(streams);

  device->algorithms().fillArray(devArray, 3.0F, ArraySize, mainStream);
  device->api().recordEventOnStream(forkEvent, mainStream);
  device->api().syncStreamWithEvent(branchStreams[0], forkEvent);
  device->algorithms().scaleArray(devArray, 4.0F, ArraySize, branchStreams[0]);
  device->api().recordEventOnStream(joinEvent, branchStreams[0]);
  device->api().syncStreamWithEvent(mainStream, joinEvent);

  device->api().streamEndCapture(graph);
  ASSERT_TRUE(graph.isInitialized());

  device->api().launchGraph(graph, mainStream);
  device->api().syncStreamWithHost(mainStream);
  for (const auto value : download()) {
    ASSERT_EQ(12.0F, value);
  }

  device->api().destroyEvent(joinEvent);
  device->api().destroyEvent(forkEvent);
}

TEST_F(Graphs, nodesRunInDependencyOrder) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  fill(0);

  auto graph = device->api().graphCreate();
  ASSERT_TRUE(graph.isInitialized());

  const auto first = device->api().graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms().fillArray(devArray, 1.0F, ArraySize, stream);
  });
  const auto second = device->api().graphAddNode(graph, {first}, mainStream, [&](void* stream) {
    device->algorithms().scaleArray(devArray, 3.0F, ArraySize, stream);
  });
  const auto third = device->api().graphAddNode(graph, {second}, mainStream, [&](void* stream) {
    device->algorithms().scaleArray(devArray, 5.0F, ArraySize, stream);
  });
  ASSERT_TRUE(third.isInitialized());

  device->api().graphInstantiate(graph);
  device->api().launchGraph(graph, mainStream);
  device->api().syncStreamWithHost(mainStream);

  // the fill has to come first: if it ran last, every entry would be 1 instead
  for (const auto value : download()) {
    ASSERT_EQ(15.0F, value);
  }
}

TEST_F(Graphs, nodesForkAndJoin) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  fill(-1);

  auto graph = device->api().graphCreate();

  const auto root = device->api().graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms().fillArray(devArray, 0.0F, ArraySize, stream);
  });

  // each branch owns a disjoint chunk and runs on its own stream, so they may overlap
  std::vector<DeviceGraphNodeHandle> branches;
  for (std::size_t i = 0; i < BranchCount; ++i) {
    branches.push_back(
        device->api().graphAddNode(graph, {root}, branchStreams[i], [&, i](void* stream) {
          device->algorithms().fillArray(
              devArray + i * ChunkSize, static_cast<float>(i + 1), ChunkSize, stream);
        }));
  }

  const auto join = device->api().graphAddNode(graph, branches, mainStream, [&](void* stream) {
    device->algorithms().scaleArray(devArray, 10.0F, ArraySize, stream);
  });
  ASSERT_TRUE(join.isInitialized());

  device->api().graphInstantiate(graph);
  device->api().launchGraph(graph, mainStream);
  device->api().syncStreamWithHost(mainStream);

  // a chunk holding i+1 means the join overtook its branch; a chunk holding 0 means the root
  // overtook it
  const auto host = download();
  for (std::size_t i = 0; i < BranchCount; ++i) {
    expectChunk(host, i, 10.0F * static_cast<float>(i + 1));
  }
}

TEST_F(Graphs, anEmptyNodeJoinsItsDependencies) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  fill(-1);

  auto graph = device->api().graphCreate();

  const auto root = device->api().graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms().fillArray(devArray, 0.0F, ArraySize, stream);
  });

  std::vector<DeviceGraphNodeHandle> branches;
  for (std::size_t i = 0; i < BranchCount; ++i) {
    branches.push_back(
        device->api().graphAddNode(graph, {root}, branchStreams[i], [&, i](void* stream) {
          device->algorithms().fillArray(
              devArray + i * ChunkSize, static_cast<float>(i + 1), ChunkSize, stream);
        }));
  }

  // a node that records nothing stands for its own dependencies, which is what makes it usable
  // as a join without costing a command
  const auto join = device->api().graphAddNode(graph, branches, mainStream, [](void*) {});

  const auto last = device->api().graphAddNode(graph, {join}, mainStream, [&](void* stream) {
    device->algorithms().scaleArray(devArray, 100.0F, ArraySize, stream);
  });
  ASSERT_TRUE(last.isInitialized());

  device->api().graphInstantiate(graph);
  device->api().launchGraph(graph, mainStream);
  device->api().syncStreamWithHost(mainStream);

  const auto host = download();
  for (std::size_t i = 0; i < BranchCount; ++i) {
    expectChunk(host, i, 100.0F * static_cast<float>(i + 1));
  }
}

TEST_F(Graphs, emptySiblingsJoinTheirSharedDependency) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  fill(-1);

  auto graph = device->api().graphCreate();

  const auto root = device->api().graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms().fillArray(devArray, 1.0F, ArraySize, stream);
  });

  // every other branch records nothing and thus stands for the root; the join then depends on
  // the root several times over, which must still name each native node only once. The other
  // branches fill rather than scale, so that the final scale has to come after them
  std::vector<DeviceGraphNodeHandle> branches;
  for (std::size_t i = 0; i < BranchCount; ++i) {
    branches.push_back(
        device->api().graphAddNode(graph, {root}, branchStreams[i], [&, i](void* stream) {
          if (i % 2 == 1) {
            device->algorithms().fillArray(
                devArray + i * ChunkSize, static_cast<float>(i + 1), ChunkSize, stream);
          }
        }));
  }

  const auto join = device->api().graphAddNode(graph, branches, mainStream, [](void*) {});

  const auto last = device->api().graphAddNode(graph, {join}, mainStream, [&](void* stream) {
    device->algorithms().scaleArray(devArray, 100.0F, ArraySize, stream);
  });
  ASSERT_TRUE(last.isInitialized());

  device->api().graphInstantiate(graph);
  device->api().launchGraph(graph, mainStream);
  device->api().syncStreamWithHost(mainStream);

  const auto host = download();
  for (std::size_t i = 0; i < BranchCount; ++i) {
    expectChunk(host, i, i % 2 == 1 ? 100.0F * static_cast<float>(i + 1) : 100.0F);
  }
}

TEST_F(Graphs, aNodeGraphCanBeLaunchedRepeatedly) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  fill(1);

  // no fill inside the graph, so repeated launches accumulate and a graph that silently ran only
  // once would be caught
  auto graph = device->api().graphCreate();
  device->api().graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms().scaleArray(devArray, 2.0F, ArraySize, stream);
  });
  device->api().graphInstantiate(graph);

  for (int i = 0; i < 3; ++i) {
    device->api().launchGraph(graph, mainStream);
  }
  device->api().syncStreamWithHost(mainStream);

  for (const auto value : download()) {
    ASSERT_EQ(8.0F, value);
  }
}

TEST_F(Graphs, isolatedNodesRunInDependencyOrder) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  fill(0);

  auto graph = device->api().graphCreate();
  const auto first = device->api().graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms().fillArray(devArray, 1.0F, ArraySize, stream);
  });

  // captured into a graph of its own, which becomes a single node
  device->api().graphBeginIsolatedNode(graph, {first}, branchStreams[0]);
  device->algorithms().scaleArray(devArray, 3.0F, ArraySize, branchStreams[0]);
  const auto second = device->api().graphEndNode(graph, branchStreams[0]);

  // one that records nothing stands for its own dependencies, as any other node
  device->api().graphBeginIsolatedNode(graph, {second}, branchStreams[1]);
  const auto empty = device->api().graphEndNode(graph, branchStreams[1]);

  const auto third = device->api().graphAddNode(graph, {empty}, mainStream, [&](void* stream) {
    device->algorithms().scaleArray(devArray, 5.0F, ArraySize, stream);
  });
  ASSERT_TRUE(third.isInitialized());

  device->api().graphInstantiate(graph);
  for (int i = 0; i < 2; ++i) {
    device->api().launchGraph(graph, mainStream);
    device->api().syncStreamWithHost(mainStream);

    // the scales run between the fill and each other: else 1, 3, 5 or 15 would come out wrong
    for (const auto value : download()) {
      ASSERT_EQ(15.0F, value);
    }
  }
}

TEST_F(Graphs, handlesOwnTheirGraph) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  DeviceGraphHandle empty;
  EXPECT_FALSE(empty.isInitialized());
  EXPECT_TRUE(!empty);

  auto graph = device->api().graphCreate();
  device->api().graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms().fillArray(devArray, 5.0F, ArraySize, stream);
  });
  device->api().graphInstantiate(graph);

  auto copy = graph;
  EXPECT_TRUE(copy.isInitialized());
  graph.reset();
  EXPECT_FALSE(graph.isInitialized());

  // the graph is still alive through the second handle
  device->api().launchGraph(copy, mainStream);
  device->api().syncStreamWithHost(mainStream);
  for (const auto value : download()) {
    ASSERT_EQ(5.0F, value);
  }
}

TEST_F(Graphs, droppedGraphsReleaseTheirResources) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  // A workload that keys its graphs on something that varies - a time step width, say - builds
  // and drops them all the time, and that has to stay flat rather than accumulate device-side
  // resources.
  for (int i = 0; i < 256; ++i) {
    auto graph = device->api().graphCreate();
    device->api().graphAddNode(graph, {}, mainStream, [&](void* stream) {
      device->algorithms().fillArray(devArray, static_cast<float>(i), ArraySize, stream);
    });
    device->api().graphInstantiate(graph);
    device->api().launchGraph(graph, mainStream);
  }
  device->api().syncStreamWithHost(mainStream);

  for (const auto value : download()) {
    ASSERT_EQ(255.0F, value);
  }
}

TEST_F(Graphs, siblingNodesMayShareAStream) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  fill(-1);

  // Two nodes with the same dependency and no edge between them, recorded onto one stream. They
  // may end up ordered - a backend that expresses edges through the recorded stream orders them -
  // but they write disjoint chunks, so the result is the same either way and neither may be lost.
  auto graph = device->api().graphCreate();

  const auto root = device->api().graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms().fillArray(devArray, 0.0F, ArraySize, stream);
  });

  std::vector<DeviceGraphNodeHandle> siblings;
  for (std::size_t i = 0; i < BranchCount; ++i) {
    siblings.push_back(device->api().graphAddNode(graph, {root}, mainStream, [&, i](void* stream) {
      device->algorithms().fillArray(
          devArray + i * ChunkSize, static_cast<float>(i + 1), ChunkSize, stream);
    }));
  }

  device->api().graphAddNode(graph, siblings, mainStream, [&](void* stream) {
    device->algorithms().scaleArray(devArray, 10.0F, ArraySize, stream);
  });

  device->api().graphInstantiate(graph);
  device->api().launchGraph(graph, mainStream);
  device->api().syncStreamWithHost(mainStream);

  const auto host = download();
  for (std::size_t i = 0; i < BranchCount; ++i) {
    expectChunk(host, i, 10.0F * static_cast<float>(i + 1));
  }
}

namespace {
// enough work in front of a graph that the graph gets launched while the work still runs
constexpr std::size_t LargeSize = std::size_t{1} << 24;
constexpr std::size_t LargeChunkSize = LargeSize / BranchCount;
constexpr int PrecedingFills = 32;
constexpr int Rounds = 4;
} // namespace

/**
 * A graph whose branches start right at its beginning has several root nodes. All of them have to
 * wait for the work enqueued on the launch stream before the graph, like a single kernel would.
 * Each root scales the chunk that the work before it fills; a root that overtakes the fills gets
 * overwritten by them afterwards.
 */
class GraphRoots : public Graphs {
  public:
  void SetUp() override {
    Graphs::SetUp();
    large = static_cast<float*>(device->api().allocGlobMem(LargeSize * sizeof(float)));
  }

  void TearDown() override {
    device->api().freeGlobMem(large);
    Graphs::TearDown();
  }

  protected:
  void scaleChunk(std::size_t chunk, void* stream) {
    device->algorithms().scaleArray(large + chunk * LargeChunkSize, 10.0F, LargeChunkSize, stream);
  }

  void launchAfterFills(DeviceGraphHandle& graph) {
    for (int round = 0; round < Rounds; ++round) {
      for (int fill = 1; fill <= PrecedingFills; ++fill) {
        device->algorithms().fillArray(large, static_cast<float>(fill), LargeSize, mainStream);
      }
      device->api().launchGraph(graph, mainStream);
      device->api().syncStreamWithHost(mainStream);

      std::vector<float> host(LargeSize, -1);
      device->api().copyFromAsync(host.data(), large, LargeSize * sizeof(float), mainStream);
      device->api().syncStreamWithHost(mainStream);
      for (std::size_t i = 0; i < LargeSize; ++i) {
        ASSERT_EQ(10.0F * PrecedingFills, host[i])
            << "at index " << i << " of chunk " << i / LargeChunkSize << " in round " << round;
      }
    }
  }

  float* large{nullptr};
};

TEST_F(GraphRoots, independentNodesWaitForTheWorkBeforeTheLaunch) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  auto graph = device->api().graphCreate();
  for (std::size_t i = 0; i < BranchCount; ++i) {
    device->api().graphAddNode(
        graph, {}, branchStreams[i], [&, i](void* stream) { scaleChunk(i, stream); });
  }
  device->api().graphInstantiate(graph);

  launchAfterFills(graph);
}

TEST_F(GraphRoots, aCaptureThatForksRightAwayWaitsForTheWorkBeforeTheLaunch) {
  if (graphCapturingUnavailable()) {
    GTEST_SKIP() << "the backend does not support graph capturing";
  }

  auto* forkEvent = device->api().createEvent();
  std::array<void*, BranchCount> joinEvents{};
  for (auto& event : joinEvents) {
    event = device->api().createEvent();
  }

  std::vector<void*> streams{mainStream};
  streams.insert(streams.end(), branchStreams.begin(), branchStreams.end());
  auto graph = device->api().streamBeginCapture(streams);
  device->api().recordEventOnStream(forkEvent, mainStream);
  for (std::size_t i = 0; i < BranchCount; ++i) {
    device->api().syncStreamWithEvent(branchStreams[i], forkEvent);
    scaleChunk(i, branchStreams[i]);
    device->api().recordEventOnStream(joinEvents[i], branchStreams[i]);
    device->api().syncStreamWithEvent(mainStream, joinEvents[i]);
  }
  device->api().streamEndCapture(graph);
  ASSERT_TRUE(graph.isInitialized());

  launchAfterFills(graph);

  for (auto* event : joinEvents) {
    device->api().destroyEvent(event);
  }
  device->api().destroyEvent(forkEvent);
}

/**
 * A graph that forks after its first node, launched again and again right after its launches
 * before: its branches have to wait for that node, and it for the work before the launch (with ROCm
 * 6.2, the branches did not wait any more after the first launch).
 */
TEST_F(GraphRoots, aGraphThatForksAfterItsFirstNodeCanBeLaunchedRightAgain) {
  if (graphCapturingUnavailable()) {
    GTEST_SKIP() << "the backend does not support graph capturing";
  }

  auto* forkEvent = device->api().createEvent();
  std::array<void*, BranchCount> joinEvents{};
  for (auto& event : joinEvents) {
    event = device->api().createEvent();
  }

  std::vector<void*> streams{mainStream};
  streams.insert(streams.end(), branchStreams.begin(), branchStreams.end());
  auto graph = device->api().streamBeginCapture(streams);
  device->algorithms().scaleArray(large, 2.0F, LargeSize, mainStream);
  device->api().recordEventOnStream(forkEvent, mainStream);
  for (std::size_t i = 0; i < BranchCount; ++i) {
    device->api().syncStreamWithEvent(branchStreams[i], forkEvent);
    device->algorithms().scaleArray(
        large + i * LargeChunkSize, 5.0F, LargeChunkSize, branchStreams[i]);
    device->api().recordEventOnStream(joinEvents[i], branchStreams[i]);
    device->api().syncStreamWithEvent(mainStream, joinEvents[i]);
  }
  device->api().streamEndCapture(graph);
  ASSERT_TRUE(graph.isInitialized());

  // the next launch follows right after the check: the host buffer is there already
  std::vector<float> host(LargeSize);
  for (int round = 0; round < 4 * Rounds; ++round) {
    for (int fill = 1; fill <= PrecedingFills; ++fill) {
      device->algorithms().fillArray(large, static_cast<float>(fill), LargeSize, mainStream);
    }
    device->api().launchGraph(graph, mainStream);
    device->api().syncStreamWithHost(mainStream);
    device->api().copyFromAsync(host.data(), large, LargeSize * sizeof(float), mainStream);
    device->api().syncStreamWithHost(mainStream);
    const auto wrong = std::count_if(
        host.begin(), host.end(), [](float value) { return value != 10.0F * PrecedingFills; });
    ASSERT_EQ(0, wrong) << "values wrong in round " << round;
  }

  for (auto* event : joinEvents) {
    device->api().destroyEvent(event);
  }
  device->api().destroyEvent(forkEvent);
}

/**
 * An event that was recorded within an earlier capture, after work there, gets recorded right at
 * the start of the next capture, where the other streams wait for it: they have to join this
 * capture (with ROCm 6.2 and 6.3, they depended on a node of the earlier graph instead, and ending
 * the capture failed).
 */
TEST_F(Graphs, anEventRecordedRightAtTheStartOfACaptureForksIt) {
  if (graphCapturingUnavailable()) {
    GTEST_SKIP() << "the backend does not support graph capturing";
  }

  auto* forkEvent = device->api().createEvent();
  std::array<void*, BranchCount> joinEvents{};
  for (auto& event : joinEvents) {
    event = device->api().createEvent();
  }

  std::vector<void*> streams{mainStream};
  streams.insert(streams.end(), branchStreams.begin(), branchStreams.end());
  for (int capture = 0; capture < 2; ++capture) {
    fill(1);
    auto graph = device->api().streamBeginCapture(streams);
    if (capture == 0) {
      // work in front of the event, so that the event stands for a node of this capture
      device->algorithms().scaleArray(devArray, 1.0F, ArraySize, mainStream);
    }
    device->api().recordEventOnStream(forkEvent, mainStream);
    for (std::size_t i = 0; i < BranchCount; ++i) {
      device->api().syncStreamWithEvent(branchStreams[i], forkEvent);
      device->algorithms().scaleArray(devArray + i * ChunkSize, 10.0F, ChunkSize, branchStreams[i]);
      device->api().recordEventOnStream(joinEvents[i], branchStreams[i]);
      device->api().syncStreamWithEvent(mainStream, joinEvents[i]);
    }
    device->api().streamEndCapture(graph);
    ASSERT_TRUE(graph.isInitialized());

    device->api().launchGraph(graph, mainStream);
    device->api().syncStreamWithHost(mainStream);
    for (const auto value : download()) {
      ASSERT_EQ(10.0F, value) << "in capture " << capture;
    }
  }

  for (auto* event : joinEvents) {
    device->api().destroyEvent(event);
  }
  device->api().destroyEvent(forkEvent);
}

TEST_F(Graphs, captureOfStreamsWaitingForEachOtherInTurn) {
  if (graphCapturingUnavailable()) {
    GTEST_SKIP() << "the backend does not support graph capturing";
  }

  fill(-1);

  // The branches wait for each other in a cycle - the first for the last - after they have joined
  // the capture, as clusters that wait for their neighbors do. Ending the capture must not trip
  // over that, and the graph has to keep the order of all the waits.
  constexpr std::size_t Branches = 3;
  auto* forkEvent = device->api().createEvent();
  std::array<void*, Branches + 1> handOver{};
  std::array<void*, Branches> joinEvents{};
  for (auto& event : handOver) {
    event = device->api().createEvent();
  }
  for (auto& event : joinEvents) {
    event = device->api().createEvent();
  }

  std::vector<void*> streams{mainStream};
  streams.insert(streams.end(), branchStreams.begin(), branchStreams.begin() + Branches);
  auto graph = device->api().streamBeginCapture(streams);
  device->api().recordEventOnStream(forkEvent, mainStream);
  for (std::size_t i = 0; i < Branches; ++i) {
    device->api().syncStreamWithEvent(branchStreams[i], forkEvent);
  }
  device->algorithms().fillArray(devArray, 1.0F, ArraySize, branchStreams[0]);
  device->api().recordEventOnStream(handOver[0], branchStreams[0]);
  for (std::size_t i = 1; i < Branches; ++i) {
    device->api().syncStreamWithEvent(branchStreams[i], handOver[i - 1]);
    device->algorithms().scaleArray(
        devArray, static_cast<float>(i + 1), ArraySize, branchStreams[i]);
    device->api().recordEventOnStream(handOver[i], branchStreams[i]);
  }
  device->api().syncStreamWithEvent(branchStreams[0], handOver[Branches - 1]);
  device->algorithms().scaleArray(devArray, 5.0F, ArraySize, branchStreams[0]);
  for (std::size_t i = 0; i < Branches; ++i) {
    device->api().recordEventOnStream(joinEvents[i], branchStreams[i]);
    device->api().syncStreamWithEvent(mainStream, joinEvents[i]);
  }
  device->api().streamEndCapture(graph);
  ASSERT_TRUE(graph.isInitialized());

  device->api().launchGraph(graph, mainStream);
  device->api().syncStreamWithHost(mainStream);
  for (const auto value : download()) {
    ASSERT_EQ(30.0F, value);
  }

  for (auto* event : joinEvents) {
    device->api().destroyEvent(event);
  }
  for (auto* event : handOver) {
    device->api().destroyEvent(event);
  }
  device->api().destroyEvent(forkEvent);
}
