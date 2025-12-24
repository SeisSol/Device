// SPDX-FileCopyrightText: 2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "CudaWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <sstream>

using namespace device;

void* ConcreteAPI::createEvent(bool withTiming) {
  cudaEvent_t event{};
  if (withTiming) {
    APIWRAP(cudaEventCreate(&event));
  } else {
    APIWRAP(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
  CHECK_ERR;
  return static_cast<void*>(event);
}

double ConcreteAPI::timespanEvents(void* eventPtrStart, void* eventPtrEnd) {
  float time{};
  cudaEvent_t start = reinterpret_cast<cudaEvent_t>(eventPtrStart);
  cudaEvent_t end = reinterpret_cast<cudaEvent_t>(eventPtrEnd);
  APIWRAP(cudaEventElapsedTime(&time, start, end));
  return static_cast<double>(time) / 1000.0;
}

void ConcreteAPI::destroyEvent(void* eventPtr) {
  cudaEvent_t event = static_cast<cudaEvent_t>(eventPtr);
  APIWRAP(cudaEventDestroy(event));
  CHECK_ERR;
}

void ConcreteAPI::syncEventWithHost(void* eventPtr) {
  cudaEvent_t event = static_cast<cudaEvent_t>(eventPtr);
  APIWRAP(cudaEventSynchronize(event));
  CHECK_ERR;
}

bool ConcreteAPI::isEventCompleted(void* eventPtr) {
  cudaEvent_t event = static_cast<cudaEvent_t>(eventPtr);
  auto result = APIWRAPX(cudaEventQuery(event), {cudaErrorNotReady});
  CHECK_ERR;
  return result == cudaSuccess;
}

void ConcreteAPI::recordEventOnHost(void* eventPtr) {
  cudaEvent_t event = static_cast<cudaEvent_t>(eventPtr);
  APIWRAP(cudaEventRecord(event));
  CHECK_ERR;
}

void ConcreteAPI::recordEventOnStream(void* eventPtr, void* streamPtr) {
  cudaEvent_t event = static_cast<cudaEvent_t>(eventPtr);
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
  APIWRAP(cudaEventRecord(event, stream));
  CHECK_ERR;
}
