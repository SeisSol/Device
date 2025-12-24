// SPDX-FileCopyrightText: 2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "hip/hip_runtime.h"
#include "HipWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <sstream>


using namespace device;

void* ConcreteAPI::createEvent() {
  hipEvent_t event;
  APIWRAP(hipEventCreate(&event));
  return static_cast<void*>(event);
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

