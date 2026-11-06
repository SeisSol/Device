// SPDX-FileCopyrightText: 2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "HipWrappedAPI.h"
#include "Internals.h"
#include "hip/hip_runtime.h"
#include "hsaqueues/HsaQueues.h"
#include "utils/logger.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <sstream>

using namespace device;

void* ConcreteAPI::createEvent(bool withTiming) {
  if (hsaqueues::enabled()) {
    return hsaqueues::createEvent(withTiming);
  }
  hipEvent_t event;
  if (withTiming) {
    APIWRAP(hipEventCreate(&event));
  } else {
    APIWRAP(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  }
  return static_cast<void*>(event);
}

double ConcreteAPI::timespanEvents(void* eventPtrStart, void* eventPtrEnd) {
  if (hsaqueues::isTagged(eventPtrStart) || hsaqueues::isTagged(eventPtrEnd)) {
    if (hsaqueues::isTagged(eventPtrStart) && hsaqueues::isTagged(eventPtrEnd)) {
      return hsaqueues::timespan(eventPtrStart, eventPtrEnd);
    }
    return 0;
  }
  float time{};
  hipEvent_t start = reinterpret_cast<hipEvent_t>(eventPtrStart);
  hipEvent_t end = reinterpret_cast<hipEvent_t>(eventPtrEnd);
  APIWRAP(hipEventElapsedTime(&time, start, end));
  return static_cast<double>(time) / 1000.0;
}

bool ConcreteAPI::isCapableOfEventTiming() {
  // HSA queues take the time only on request: by a kernel for each event with timing
  return !hsaqueues::enabled() || hsaqueues::timing();
}

void ConcreteAPI::destroyEvent(void* eventPtr) {
  if (hsaqueues::isTagged(eventPtr)) {
    hsaqueues::destroyEvent(eventPtr);
    return;
  }
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  internals::forgetEvent(event);
  APIWRAP(hipEventDestroy(event));
}

void ConcreteAPI::syncEventWithHost(void* eventPtr) {
  if (hsaqueues::isTagged(eventPtr)) {
    hsaqueues::syncEvent(eventPtr);
    return;
  }
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);

  CHECK_ERR;

  APIWRAP(hipEventSynchronize(event));
}

bool ConcreteAPI::isEventCompleted(void* eventPtr) {
  if (hsaqueues::isTagged(eventPtr)) {
    return hsaqueues::eventDone(eventPtr);
  }
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  auto result = APIWRAPX(hipEventQuery(event), {hipErrorNotReady});
  return result == hipSuccess;
}

void ConcreteAPI::recordEventOnHost(void* eventPtr) {
  if (hsaqueues::isTagged(eventPtr)) {
    hsaqueues::recordEventOnHost(eventPtr);
    return;
  }
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  internals::forgetEvent(event);
  APIWRAP(hipEventRecord(event));
}

void ConcreteAPI::recordEventOnStream(void* eventPtr, void* streamPtr) {
  if (hsaqueues::isTagged(eventPtr)) {
    hsaqueues::recordEvent(eventPtr, streamPtr);
    return;
  }
  if (hsaqueues::isTagged(streamPtr)) {
    logError() << "An event of HIP cannot be recorded on an HSA stream; create the events after "
                  "enabling HSA queues.";
  }
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
  internals::anchorCapture(stream);
  APIWRAP(hipEventRecord(event, stream));
  internals::noteRecordedEvent(event, stream);
}
