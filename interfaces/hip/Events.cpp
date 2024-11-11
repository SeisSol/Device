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
  hipEventCreate(&event);
  CHECK_ERR;
  return static_cast<void*>(event);
}

void ConcreteAPI::destroyEvent(void* eventPtr) {
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  hipEventDestroy(event);
  CHECK_ERR;
}

void ConcreteAPI::syncEventWithHost(void* eventPtr) {
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  hipEventSynchronize(event);
  CHECK_ERR;
}

bool ConcreteAPI::isEventCompleted(void* eventPtr) {
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  auto result = hipEventQuery(event);
  CHECK_ERR;
  return result == hipSuccess;
}

void ConcreteAPI::recordEventOnHost(void* eventPtr) {
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  hipEventRecord(event);
  CHECK_ERR;
}

void ConcreteAPI::recordEventOnStream(void* eventPtr, void* streamPtr) {
  hipEvent_t event = static_cast<hipEvent_t>(eventPtr);
  hipStream_t stream = static_cast<hipStream_t>(streamPtr);
  hipEventRecord(event, stream);
  CHECK_ERR;
}

