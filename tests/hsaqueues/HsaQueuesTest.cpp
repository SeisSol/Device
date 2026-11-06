// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

// Tests the HSA queues of the Device library (HIP): kernels, copies, events across queues, graphs
// built node by node (with temporary allocations), host functions, the mix with HIP streams, and
// the host time of launches.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <device.h>
#include <hip/hip_runtime.h>
#include <thread>
#include <utility>
#include <vector>

namespace {
__global__ void addValue(float* data, float value, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] += value;
  }
}

__global__ void scaleInto(const float* in, float* out, float factor, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * factor;
  }
}

// slow on purpose: a queue that waits for it has to wait for real
__global__ void slowAdd(float* data, float value, int n, int spins) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = data[i];
    for (int k = 0; k < spins; ++k) {
      x = x * 1.0000001f;
    }
    data[i] = x - x + data[i] + value;
  }
}

int failures = 0;
void expect(bool ok, const char* what) {
  std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
  failures += ok ? 0 : 1;
}
} // namespace

int main() {
  // the test qualifies new architectures as well (see README.md)
  setenv("DEVICE_HSA_QUEUE_UNTESTED", "1", 0);
  auto& device = device::DeviceInstance::instance();
  auto& api = device.api();
  api.setDevice(0);
  api.initialize();
  if (!api.enableHsaQueues()) {
    std::printf("HSA queues not available\n");
    return 1;
  }
  const int n = 1 << 16;
  const auto bytes = n * sizeof(float);
  auto* a = static_cast<float*>(api.allocGlobMem(bytes));
  auto* b = static_cast<float*>(api.allocGlobMem(bytes));
  auto* pinned = static_cast<float*>(api.allocPinnedMem(bytes, false, device::Destination::Host));
  std::vector<float> pageable(n);

  void* s1 = api.createStreamOnHsaQueue(1);
  void* s2 = api.createStreamOnHsaQueue(2);
  void* s1b = api.createStreamOnHsaQueue(1);
  void* hipStream = api.createStream();

  // (1) kernels and copies on one HSA stream
  for (int i = 0; i < n; ++i) {
    pinned[i] = static_cast<float>(i);
  }
  api.copyToAsync(a, pinned, bytes, s1);
  addValue<<<n / 256, 256, 0, static_cast<hipStream_t>(s1)>>>(a, 1.0f, n);
  api.copyFromAsync(pinned, a, bytes, s1);
  api.syncStreamWithHost(s1);
  bool ok = true;
  for (int i = 0; i < n; ++i) {
    ok = ok && pinned[i] == static_cast<float>(i) + 1.0f;
  }
  expect(ok, "copy to, kernel, copy from on an HSA stream");

  // pageable copies
  for (int i = 0; i < n; ++i) {
    pageable[i] = 2.0f * i;
  }
  api.copyToAsync(b, pageable.data(), bytes, s2);
  api.copyBetweenAsync(a, b, bytes, s2);
  std::vector<float> back(n);
  api.copyFromAsync(back.data(), a, bytes, s2);
  api.syncStreamWithHost(s2);
  ok = true;
  for (int i = 0; i < n; ++i) {
    ok = ok && back[i] == 2.0f * i;
  }
  expect(ok, "pageable copies and a device copy on another HSA stream");

  // (2) events across queues: s2 waits for a slow kernel on s1
  void* event = api.createEvent(false);
  api.copyBetweenAsync(b, a, bytes, s1);
  slowAdd<<<n / 256, 256, 0, static_cast<hipStream_t>(s1)>>>(b, 5.0f, n, 20000);
  api.recordEventOnStream(event, s1);
  api.syncStreamWithEvent(s2, event);
  scaleInto<<<n / 256, 256, 0, static_cast<hipStream_t>(s2)>>>(b, a, 0.5f, n);
  api.copyFromAsync(pinned, a, bytes, s2);
  api.syncStreamWithHost(s2);
  ok = true;
  for (int i = 0; i < n; ++i) {
    ok = ok && pinned[i] == (2.0f * i + 5.0f) * 0.5f;
  }
  expect(ok, "a stream waits for an event of another queue");
  expect(api.isEventCompleted(event), "the event has completed");

  // a HIP stream waits for an HSA event, and an HSA stream for a HIP event
  void* event2 = api.createEvent(false);
  addValue<<<n / 256, 256, 0, static_cast<hipStream_t>(s1)>>>(a, 1.0f, n);
  api.recordEventOnStream(event2, s1);
  api.syncStreamWithEvent(hipStream, event2);
  addValue<<<n / 256, 256, 0, static_cast<hipStream_t>(hipStream)>>>(a, 1.0f, n);
  api.recordEventOnStream(event2, hipStream);
  api.syncStreamWithEvent(s2, event2);
  api.copyFromAsync(pinned, a, bytes, s2);
  api.syncStreamWithHost(s2);
  ok = true;
  for (int i = 0; i < n; ++i) {
    ok = ok && pinned[i] == (2.0f * i + 5.0f) * 0.5f + 2.0f;
  }
  expect(ok, "HSA and HIP streams wait for each other");

  // (3) a graph built node by node on an HSA stream, with a temporary allocation, replayed
  auto graph = api.graphCreate();
  api.graphBeginNode(graph, {}, s1);
  auto* tmp = static_cast<float*>(api.allocMemAsync(bytes, s1));
  scaleInto<<<n / 256, 256, 0, static_cast<hipStream_t>(s1)>>>(a, tmp, 2.0f, n);
  addValue<<<n / 256, 256, 0, static_cast<hipStream_t>(s1)>>>(tmp, 3.0f, n);
  scaleInto<<<n / 256, 256, 0, static_cast<hipStream_t>(s1)>>>(tmp, b, 1.0f, n);
  api.freeMemAsync(tmp, s1);
  auto first = api.graphEndNode(graph, s1);
  // two branches after it
  api.graphBeginNode(graph, {first}, s1);
  addValue<<<n / 256, 256, 0, static_cast<hipStream_t>(s1)>>>(b, 1.0f, n);
  auto left = api.graphEndNode(graph, s1);
  api.graphBeginNode(graph, {first}, hipStream);
  addValue<<<n / 256, 256, 0, static_cast<hipStream_t>(hipStream)>>>(a, 10.0f, n);
  auto right = api.graphEndNode(graph, hipStream);
  api.graphBeginNode(graph, {left, right}, s1);
  api.graphEndNode(graph, s1);
  api.graphInstantiate(graph);
  // a = x; per replay: b = 2a + 3 + 1, a += 10
  api.copyFromAsync(pinned, a, bytes, s1);
  api.syncStreamWithHost(s1);
  std::vector<float> start(pinned, pinned + n);
  const int replays = 5;
  for (int r = 0; r < replays; ++r) {
    api.launchGraph(graph, s1);
  }
  api.copyFromAsync(pinned, b, bytes, s1);
  std::vector<float> resultB(n);
  api.syncStreamWithHost(s1);
  std::memcpy(resultB.data(), pinned, bytes);
  api.copyFromAsync(pinned, a, bytes, s1);
  api.syncStreamWithHost(s1);
  ok = true;
  for (int i = 0; i < n; ++i) {
    const float aBefore = start[i] + 10.0f * (replays - 1);
    ok = ok && resultB[i] == 2.0f * aBefore + 4.0f && pinned[i] == start[i] + 10.0f * replays;
  }
  expect(ok, "a graph with a temporary allocation and branches, replayed on an HSA stream");

  // the same graph on a HIP stream
  api.launchGraph(graph, hipStream);
  api.syncStreamWithHost(hipStream);
  api.copyFromAsync(pinned, a, bytes, s1);
  api.syncStreamWithHost(s1);
  ok = true;
  for (int i = 0; i < n; ++i) {
    ok = ok && pinned[i] == start[i] + 10.0f * (replays + 1);
  }
  expect(ok, "the same graph on a stream of HIP");

  // (4) host functions: the queue waits for them
  std::atomic<int> calls{0};
  float seen = 0;
  addValue<<<n / 256, 256, 0, static_cast<hipStream_t>(s2)>>>(a, 1.0f, n);
  api.copyFromAsync(pinned, a, sizeof(float), s2);
  api.streamHostFunction(s2, [&]() {
    seen = pinned[0];
    ++calls;
  });
  api.copyToAsync(a, pageable.data(), sizeof(float), s2);
  api.streamHostFunction(s1b, [&]() { ++calls; });
  api.syncStreamWithHost(s2);
  api.syncStreamWithHost(s1b);
  expect(calls.load() == 2 && seen == start[0] + 10.0f * (replays + 1) + 1.0f,
         "host functions run in stream order");

  // (5) many launches: the ring of kernel arguments wraps around
  for (int r = 0; r < 200000; ++r) {
    addValue<<<1, 64, 0, static_cast<hipStream_t>(s1b)>>>(b, 1.0f, 64);
  }
  api.copyFromAsync(pinned, b, 64 * sizeof(float), s1b);
  api.syncDevice();
  ok = true;
  for (int i = 0; i < 64; ++i) {
    ok = ok && pinned[i] == 2.0f * (start[i] + 10.0f * replays) + 4.0f + 200000.0f;
  }
  expect(ok, "200000 launches on one stream");

  // (5b) the host never waits to enqueue: a stream waits for a host counter, and the host enqueues
  // more kernels than its queue holds before it sets the counter
  {
    auto* counter = static_cast<std::uint32_t*>(
        api.allocPinnedMem(sizeof(std::uint32_t), false, device::Destination::CurrentDevice));
    *counter = 0;
    api.copyFromAsync(pinned, b, 64 * sizeof(float), s1b);
    api.syncStreamWithHost(s1b);
    const float before = pinned[0];
    api.streamWaitMemory(s1b, counter, 1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 300000; ++r) {
      addValue<<<1, 64, 0, static_cast<hipStream_t>(s1b)>>>(b, 1.0f, 64);
    }
    const double enqueue =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    __atomic_store_n(counter, 1U, __ATOMIC_RELEASE);
    api.copyFromAsync(pinned, b, 64 * sizeof(float), s1b);
    api.syncStreamWithHost(s1b);
    expect(pinned[0] == before + 300000.0f,
           "300000 launches behind a wait for the host, enqueued without waiting");
    std::printf("      (enqueued in %.2f s)\n", enqueue);
    api.freePinnedMem(counter);
  }

  // (5c) a counter of the device API (a signal with HSA queues): the command processor waits for
  // it; the host stores into it like into memory
  {
    auto* counter = api.allocCounter();
    auto* counterOnDevice = static_cast<std::uint32_t*>(api.devicePointer(counter));
    api.copyFromAsync(pinned, b, 64 * sizeof(float), s2);
    api.syncStreamWithHost(s2);
    const float before = pinned[0];
    api.streamWaitMemory(s2, counter, 3);
    addValue<<<1, 64, 0, static_cast<hipStream_t>(s2)>>>(b, 1.0f, 64);
    api.copyFromAsync(pinned, b, 64 * sizeof(float), s2);
    __atomic_store_n(counter, 2U, __ATOMIC_RELEASE);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const bool waited = !api.isStreamWorkDone(s2);
    const auto t0 = std::chrono::steady_clock::now();
    __atomic_store_n(counter, 3U, __ATOMIC_RELEASE);
    api.syncStreamWithHost(s2);
    const double wake =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    expect(waited && pinned[0] == before + 1.0f && counterOnDevice != nullptr,
           "a stream waits for a counter of the host until it is reached");
    std::printf("      (%.1f us from the store to the end of the stream)\n", wake);
    api.freeCounter(counter);
  }

  // (6) host time of launches: HSA stream vs stream of HIP, kernels and the graph
  for (int round = 0; round < 2; ++round) {
    const int count = 20000;
    const auto time = [&](void* stream, bool graphs) {
      api.syncDevice();
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < count; ++r) {
        if (graphs) {
          api.launchGraph(graph, stream);
        } else {
          addValue<<<1, 64, 0, static_cast<hipStream_t>(stream)>>>(b, 1.0f, 64);
        }
      }
      const auto t1 = std::chrono::steady_clock::now();
      api.syncDevice();
      const auto t2 = std::chrono::steady_clock::now();
      return std::make_pair(std::chrono::duration<double, std::micro>(t1 - t0).count() / count,
                            std::chrono::duration<double, std::micro>(t2 - t0).count() / count);
    };
    const auto hsaKernels = time(s1b, false);
    const auto hipKernels = time(hipStream, false);
    const auto hsaGraphs = time(s1b, true);
    const auto hipGraphs = time(hipStream, true);
    std::printf("round %d: kernel launch host/total us: HSA %.2f/%.2f, HIP %.2f/%.2f; graph (5 "
                "kernels) HSA %.2f/%.2f, HIP %.2f/%.2f\n",
                round,
                hsaKernels.first,
                hsaKernels.second,
                hipKernels.first,
                hipKernels.second,
                hsaGraphs.first,
                hsaGraphs.second,
                hipGraphs.first,
                hipGraphs.second);
  }

  // (7) device time of large kernels (many workgroups read the kernel arguments)
  {
    const int big = 1 << 26;
    auto* data = static_cast<float*>(api.allocGlobMem(big * sizeof(float)));
    for (int round = 0; round < 2; ++round) {
      const auto time = [&](void* stream) {
        api.syncDevice();
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 100; ++r) {
          addValue<<<big / 256, 256, 0, static_cast<hipStream_t>(stream)>>>(data, 1.0f, big);
        }
        api.syncDevice();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                   .count() /
               100;
      };
      const double hsa = time(s1b);
      const double hip = time(hipStream);
      std::printf("round %d: large kernel (%d blocks) us: HSA %.1f, HIP %.1f\n",
                  round,
                  big / 256,
                  hsa,
                  hip);
    }
    api.freeGlobMem(data);
  }

  api.destroyEvent(event);
  api.destroyEvent(event2);
  api.destroyGenericStream(s1);
  api.destroyGenericStream(s2);
  api.destroyGenericStream(s1b);
  api.destroyGenericStream(hipStream);
  api.freeGlobMem(a);
  api.freeGlobMem(b);
  api.freePinnedMem(pinned);
  std::printf("%d failures\n", failures);
  device.finalize();
  return failures == 0 ? 0 : 1;
}
