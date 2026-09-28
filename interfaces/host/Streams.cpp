// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DataTypes.h"
#include "HostWrappedAPI.h"
#include "utils/logger.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using namespace device;

ConcreteAPI::Event* ConcreteAPI::toEvent(void* eventPtr) { return static_cast<Event*>(eventPtr); }

void* ConcreteAPI::getDefaultStream() {
  isFlagSet<InterfaceInitialized>(status);
  return &defaultStream;
}

void ConcreteAPI::syncDefaultStreamWithHost() { isFlagSet<InterfaceInitialized>(status); }

void* ConcreteAPI::createStream(double /*priority*/) {
  isFlagSet<InterfaceInitialized>(status);
  auto stream = std::make_unique<Stream>();
  void* streamPtr = stream.get();
  std::lock_guard guard(apiMutex);
  genericStreams.emplace(streamPtr, std::move(stream));
  return streamPtr;
}

void ConcreteAPI::destroyGenericStream(void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  std::lock_guard guard(apiMutex);
  genericStreams.erase(streamPtr);
}

void ConcreteAPI::syncStreamWithHost(void* /*streamPtr*/) {
  isFlagSet<InterfaceInitialized>(status);
}

bool ConcreteAPI::isStreamWorkDone(void* /*streamPtr*/) {
  isFlagSet<InterfaceInitialized>(status);
  return true;
}

void ConcreteAPI::syncStreamWithEvent(void* /*streamPtr*/, void* /*eventPtr*/) {}

void ConcreteAPI::streamHostFunction(void* /*streamPtr*/, const std::function<void()>& function) {
  // everything enqueued before has completed already; thus, the function can run right away
  function();
}

void ConcreteAPI::streamWaitMemory(void* /*streamPtr*/, uint32_t* location, uint32_t value) {
  // the stream runs on the calling thread; hence, it is the caller that waits until another
  // thread has written the value
  while (__atomic_load_n(location, __ATOMIC_ACQUIRE) < value) {
    std::this_thread::yield();
  }
}

bool ConcreteAPI::isCapableOfGraphCapturing() { return false; }

DeviceGraphHandle ConcreteAPI::streamBeginCapture(std::vector<void*>& /*streamPtrs*/) {
  return DeviceGraphHandle();
}

void ConcreteAPI::streamEndCapture(DeviceGraphHandle /*handle*/) {}

void ConcreteAPI::launchGraph(DeviceGraphHandle /*graphHandle*/, void* /*streamPtr*/) {}

void* ConcreteAPI::createEvent(bool /*withTiming*/) { return new Event(); }

void ConcreteAPI::destroyEvent(void* eventPtr) { delete toEvent(eventPtr); }

void ConcreteAPI::syncEventWithHost(void* /*eventPtr*/) {}

bool ConcreteAPI::isEventCompleted(void* /*eventPtr*/) { return true; }

void ConcreteAPI::recordEventOnHost(void* eventPtr) {
  toEvent(eventPtr)->recorded = std::chrono::steady_clock::now();
}

void ConcreteAPI::recordEventOnStream(void* eventPtr, void* /*streamPtr*/) {
  toEvent(eventPtr)->recorded = std::chrono::steady_clock::now();
}

double ConcreteAPI::timespanEvents(void* eventPtrStart, void* eventPtrEnd) {
  const auto span = toEvent(eventPtrEnd)->recorded - toEvent(eventPtrStart)->recorded;
  return std::chrono::duration<double>(span).count();
}
