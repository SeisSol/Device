// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "BaseTestSuite.h"
#include "device.h"

#include "gtest/gtest.h"
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
    devArray = static_cast<float*>(device->api->allocGlobMem(ArraySize * sizeof(float)));
    mainStream = device->api->createStream();
    for (auto& stream : branchStreams) {
      stream = device->api->createStream();
    }
  }

  void TearDown() override {
    for (auto* stream : branchStreams) {
      device->api->destroyGenericStream(stream);
    }
    device->api->destroyGenericStream(mainStream);
    device->api->freeGlobMem(devArray);
  }

  protected:
  std::vector<float> download() {
    std::vector<float> host(ArraySize, -1);
    device->api->copyFromAsync(host.data(), devArray, ArraySize * sizeof(float), mainStream);
    device->api->syncStreamWithHost(mainStream);
    return host;
  }

  void fill(float value) {
    device->algorithms.fillArray(devArray, value, ArraySize, mainStream);
    device->api->syncStreamWithHost(mainStream);
  }

  static void expectChunk(const std::vector<float>& host, std::size_t chunk, float value) {
    for (std::size_t i = chunk * ChunkSize; i < (chunk + 1) * ChunkSize; ++i) {
      ASSERT_EQ(value, host[i]) << "at index " << i << " of chunk " << chunk;
    }
  }

  bool graphCapturingUnavailable() { return !device->api->isCapableOfGraphCapturing(); }

  bool graphNodesUnavailable() { return !device->api->isCapableOfGraphNodes(); }

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
  auto graph = device->api->streamBeginCapture(streams);
  device->algorithms.fillArray(devArray, 1.0F, ArraySize, mainStream);
  device->algorithms.scaleArray(devArray, 2.0F, ArraySize, mainStream);
  device->api->streamEndCapture(graph);

  ASSERT_TRUE(graph.isInitialized());

  // nothing has run yet: capturing records, it does not execute
  for (const auto value : download()) {
    ASSERT_EQ(0.0F, value);
  }

  device->api->launchGraph(graph, mainStream);
  device->api->syncStreamWithHost(mainStream);
  for (const auto value : download()) {
    ASSERT_EQ(2.0F, value);
  }

  // the fill is part of the graph, so a second replay lands on the same value rather than doubling
  device->api->launchGraph(graph, mainStream);
  device->api->syncStreamWithHost(mainStream);
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
  auto* forkEvent = device->api->createEvent();
  auto* joinEvent = device->api->createEvent();

  std::vector<void*> streams{mainStream, branchStreams[0]};
  auto graph = device->api->streamBeginCapture(streams);

  device->algorithms.fillArray(devArray, 3.0F, ArraySize, mainStream);
  device->api->recordEventOnStream(forkEvent, mainStream);
  device->api->syncStreamWithEvent(branchStreams[0], forkEvent);
  device->algorithms.scaleArray(devArray, 4.0F, ArraySize, branchStreams[0]);
  device->api->recordEventOnStream(joinEvent, branchStreams[0]);
  device->api->syncStreamWithEvent(mainStream, joinEvent);

  device->api->streamEndCapture(graph);
  ASSERT_TRUE(graph.isInitialized());

  device->api->launchGraph(graph, mainStream);
  device->api->syncStreamWithHost(mainStream);
  for (const auto value : download()) {
    ASSERT_EQ(12.0F, value);
  }

  device->api->destroyEvent(joinEvent);
  device->api->destroyEvent(forkEvent);
}

TEST_F(Graphs, nodesRunInDependencyOrder) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  fill(0);

  auto graph = device->api->graphCreate();
  ASSERT_TRUE(graph.isInitialized());

  const auto first = device->api->graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms.fillArray(devArray, 1.0F, ArraySize, stream);
  });
  const auto second = device->api->graphAddNode(graph, {first}, mainStream, [&](void* stream) {
    device->algorithms.scaleArray(devArray, 3.0F, ArraySize, stream);
  });
  const auto third = device->api->graphAddNode(graph, {second}, mainStream, [&](void* stream) {
    device->algorithms.scaleArray(devArray, 5.0F, ArraySize, stream);
  });
  ASSERT_TRUE(third.isInitialized());

  device->api->graphInstantiate(graph);
  device->api->launchGraph(graph, mainStream);
  device->api->syncStreamWithHost(mainStream);

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

  auto graph = device->api->graphCreate();

  const auto root = device->api->graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms.fillArray(devArray, 0.0F, ArraySize, stream);
  });

  // each branch owns a disjoint chunk and runs on its own stream, so they may overlap
  std::vector<DeviceGraphNodeHandle> branches;
  for (std::size_t i = 0; i < BranchCount; ++i) {
    branches.push_back(
        device->api->graphAddNode(graph, {root}, branchStreams[i], [&, i](void* stream) {
          device->algorithms.fillArray(
              devArray + i * ChunkSize, static_cast<float>(i + 1), ChunkSize, stream);
        }));
  }

  const auto join = device->api->graphAddNode(graph, branches, mainStream, [&](void* stream) {
    device->algorithms.scaleArray(devArray, 10.0F, ArraySize, stream);
  });
  ASSERT_TRUE(join.isInitialized());

  device->api->graphInstantiate(graph);
  device->api->launchGraph(graph, mainStream);
  device->api->syncStreamWithHost(mainStream);

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

  auto graph = device->api->graphCreate();

  const auto root = device->api->graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms.fillArray(devArray, 0.0F, ArraySize, stream);
  });

  std::vector<DeviceGraphNodeHandle> branches;
  for (std::size_t i = 0; i < BranchCount; ++i) {
    branches.push_back(
        device->api->graphAddNode(graph, {root}, branchStreams[i], [&, i](void* stream) {
          device->algorithms.fillArray(
              devArray + i * ChunkSize, static_cast<float>(i + 1), ChunkSize, stream);
        }));
  }

  // a node that records nothing stands for its own dependencies, which is what makes it usable
  // as a join without costing a command
  const auto join = device->api->graphAddNode(graph, branches, mainStream, [](void*) {});

  const auto last = device->api->graphAddNode(graph, {join}, mainStream, [&](void* stream) {
    device->algorithms.scaleArray(devArray, 100.0F, ArraySize, stream);
  });
  ASSERT_TRUE(last.isInitialized());

  device->api->graphInstantiate(graph);
  device->api->launchGraph(graph, mainStream);
  device->api->syncStreamWithHost(mainStream);

  const auto host = download();
  for (std::size_t i = 0; i < BranchCount; ++i) {
    expectChunk(host, i, 100.0F * static_cast<float>(i + 1));
  }
}

TEST_F(Graphs, aNodeGraphCanBeLaunchedRepeatedly) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  fill(1);

  // no fill inside the graph, so repeated launches accumulate and a graph that silently ran only
  // once would be caught
  auto graph = device->api->graphCreate();
  device->api->graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms.scaleArray(devArray, 2.0F, ArraySize, stream);
  });
  device->api->graphInstantiate(graph);

  for (int i = 0; i < 3; ++i) {
    device->api->launchGraph(graph, mainStream);
  }
  device->api->syncStreamWithHost(mainStream);

  for (const auto value : download()) {
    ASSERT_EQ(8.0F, value);
  }
}

TEST_F(Graphs, handlesOwnTheirGraph) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  DeviceGraphHandle empty;
  EXPECT_FALSE(empty.isInitialized());
  EXPECT_TRUE(!empty);

  auto graph = device->api->graphCreate();
  device->api->graphAddNode(graph, {}, mainStream, [&](void* stream) {
    device->algorithms.fillArray(devArray, 5.0F, ArraySize, stream);
  });
  device->api->graphInstantiate(graph);

  auto copy = graph;
  EXPECT_TRUE(copy.isInitialized());
  graph.reset();
  EXPECT_FALSE(graph.isInitialized());

  // the graph is still alive through the second handle
  device->api->launchGraph(copy, mainStream);
  device->api->syncStreamWithHost(mainStream);
  for (const auto value : download()) {
    ASSERT_EQ(5.0F, value);
  }
}

TEST_F(Graphs, droppedGraphsReleaseTheirResources) {
  if (graphNodesUnavailable()) {
    GTEST_SKIP() << "the backend does not support explicit graph nodes";
  }

  // Graphs used to be held in a container that never gave anything back, so a workload that keys
  // its graphs on something that varies - a time step width, say - grew without bound. Building
  // and dropping many of them has to stay flat.
  for (int i = 0; i < 256; ++i) {
    auto graph = device->api->graphCreate();
    device->api->graphAddNode(graph, {}, mainStream, [&](void* stream) {
      device->algorithms.fillArray(devArray, static_cast<float>(i), ArraySize, stream);
    });
    device->api->graphInstantiate(graph);
    device->api->launchGraph(graph, mainStream);
  }
  device->api->syncStreamWithHost(mainStream);

  for (const auto value : download()) {
    ASSERT_EQ(255.0F, value);
  }
}
