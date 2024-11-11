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

void* ConcreteAPI::createEvent() {
  cudaEvent_t event;
  cudaEventCreate(&event);
  CHECK_ERR;
  return static_cast<void*>(event);
}

void ConcreteAPI::destroyEvent(void* eventPtr) {
  cudaEvent_t event = static_cast<cudaEvent_t>(eventPtr);
  cudaEventDestroy(event);
  CHECK_ERR;
}

void ConcreteAPI::syncEventWithHost(void* eventPtr) {
  cudaEvent_t event = static_cast<cudaEvent_t>(eventPtr);
  cudaEventSynchronize(event);
  CHECK_ERR;
}

bool ConcreteAPI::isEventCompleted(void* eventPtr) {
  cudaEvent_t event = static_cast<cudaEvent_t>(eventPtr);
  auto result = cudaEventQuery(event);
  CHECK_ERR;
  return result == cudaSuccess;
}

void ConcreteAPI::recordEventOnHost(void* eventPtr) {
  cudaEvent_t event = static_cast<cudaEvent_t>(eventPtr);
  cudaEventRecord(event);
  CHECK_ERR;
}

void ConcreteAPI::recordEventOnStream(void* eventPtr, void* streamPtr) {
  cudaEvent_t event = static_cast<cudaEvent_t>(eventPtr);
  cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
  cudaEventRecord(event, stream);
  CHECK_ERR;
}

