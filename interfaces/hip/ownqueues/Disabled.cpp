// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

// The interface of own queues in a build without them (DEVICE_HIP_OWN_QUEUES=OFF): they never get
// enabled, so no handle of an own stream or event ever exists, and nothing but enable(), enabled(),
// registerLaunchHook() and the other calls that make sense without them gets called.

#include "OwnQueues.h"
#include "utils/logger.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>

namespace device::ownqueues {

namespace {
[[noreturn]] void unavailable(const char* call) {
  logError() << "Own queues are not built (DEVICE_HIP_OWN_QUEUES=OFF), but" << call
             << "got called.";
  std::abort();
}
} // namespace

struct ProgramsImpl {};

Programs::Programs() = default;
Programs::~Programs() = default;

void registerLaunchHook(LaunchFunction /*hipLaunch*/) {}

bool enable() {
  logWarning() << "Own queues are not built (DEVICE_HIP_OWN_QUEUES=OFF); the streams stay those of "
                  "HIP.";
  return false;
}
bool enabled() { return false; }

void* createStream(int /*queue*/, double /*priority*/) { unavailable("createStream"); }
void destroyStream(void* /*stream*/) { unavailable("destroyStream"); }
hipStream_t vehicle(void* stream) { return static_cast<hipStream_t>(stream); }
bool capturing(void* /*stream*/) { return false; }
void beginCapture(void* /*stream*/) {}
void endCapture(void* /*stream*/) {}
hipError_t launch(const void* /*function*/,
                  dim3 /*grid*/,
                  dim3 /*block*/,
                  void** /*args*/,
                  std::size_t /*sharedMemBytes*/,
                  hipStream_t /*stream*/) {
  unavailable("launch");
}
void syncStream(void* /*stream*/) { unavailable("syncStream"); }
bool streamDone(void* /*stream*/) { unavailable("streamDone"); }
void hostFunction(void* /*stream*/, const std::function<void()>& /*function*/) {
  unavailable("hostFunction");
}
void waitMemory(void* /*stream*/, std::uint32_t* /*location*/, std::uint32_t /*value*/) {
  unavailable("waitMemory");
}
std::uint32_t* allocCounter() { unavailable("allocCounter"); }
void freeCounter(std::uint32_t* /*counter*/) { unavailable("freeCounter"); }
bool isCounter(const void* /*location*/) { return false; }
void copy(void* /*dst*/, const void* /*src*/, std::size_t /*bytes*/, void* /*stream*/) {
  unavailable("copy");
}
void* allocAsync(std::size_t /*bytes*/, void* /*stream*/) { unavailable("allocAsync"); }
void freeAsync(void* /*ptr*/, void* /*stream*/) { unavailable("freeAsync"); }
void* createEvent(bool /*withTiming*/) { unavailable("createEvent"); }
void destroyEvent(void* /*event*/) { unavailable("destroyEvent"); }
hipEvent_t hipEventOf(void* event) { return static_cast<hipEvent_t>(event); }
void recordEvent(void* /*event*/, void* /*stream*/) { unavailable("recordEvent"); }
void recordEventOnHost(void* /*event*/) { unavailable("recordEventOnHost"); }
void streamWaitEvent(void* /*stream*/, void* /*event*/) { unavailable("streamWaitEvent"); }
double timespan(void* /*start*/, void* /*end*/) { unavailable("timespan"); }
bool timing() { return false; }
bool trace(const TraceSink& /*sink*/) { return false; }
void nameStream(void* /*stream*/, const std::string& /*name*/) {}
bool eventDone(void* /*event*/) { unavailable("eventDone"); }
void syncEvent(void* /*event*/) { unavailable("syncEvent"); }
void launchGraph(Programs& /*programs*/, hipGraph_t /*graph*/, void* /*stream*/) {
  unavailable("launchGraph");
}
bool noteOrigin(Programs& /*programs*/, hipGraphNode_t /*node*/, void* /*stream*/) { return false; }
void syncAll() {}
void finalize() {}

} // namespace device::ownqueues
