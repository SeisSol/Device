// SPDX-FileCopyrightText: 2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "HipWrappedAPI.h"
#include "Internals.h"
#include "hip/hip_runtime.h"
#include "utils/logger.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <sstream>

using namespace device;

void* ConcreteAPI::createEvent(bool withTiming) {
  hipEvent_t event;
  if (withTiming) {
    APIWRAP(hipEventCreate(&event));
  } else {
    APIWRAP(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  }
  return static_cast<void*>(event);
}

double ConcreteAPI::timespanEvents(void* eventPtrStart, void* eventPtrEnd) {
  float time{};
  hipEvent_t start = reinterpret_cast<hipEvent_t>(eventPtrStart);
  hipEvent_t end = reinterpret_cast<hipEvent_t>(eventPtrEnd);
  APIWRAP(hipEventElapsedTime(&time, start, end));
  return static_cast<double>(time) / 1000.0;
}

void ConcreteAPI::destroyEvent(void* eventPtr) {
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  APIWRAP(hipEventDestroy(event));
}

void ConcreteAPI::syncEventWithHost(void* eventPtr) {
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  APIWRAP(hipEventSynchronize(event));
}

bool ConcreteAPI::isEventCompleted(void* eventPtr) {
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  auto result = APIWRAPX(hipEventQuery(event), {hipErrorNotReady});
  return result == hipSuccess;
}

void ConcreteAPI::recordEventOnHost(void* eventPtr) {
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  APIWRAP(hipEventRecord(event));
}

void ConcreteAPI::recordEventOnStream(void* eventPtr, void* streamPtr) {
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
  APIWRAP(hipEventRecord(event, stream));
}
