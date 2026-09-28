// SPDX-FileCopyrightText: 2023 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "SyclWrappedAPI.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#ifdef ONEAPI_UNDERHOOD
#include <sycl/queue.hpp>
#endif // ONEAPI_UNDERHOOD

#include <sycl/sycl.hpp>

using namespace device;

void* ConcreteAPI::getDefaultStream() { return &(this->currentQueueBuffer().getDefaultQueue()); }

void ConcreteAPI::syncDefaultStreamWithHost() {
  auto& defaultQueue = this->currentQueueBuffer().getDefaultQueue();
  this->currentQueueBuffer().syncQueueWithHost(&defaultQueue);
}

void* ConcreteAPI::createStream(double priority) {
  return this->currentQueueBuffer().newQueue(priority);
}

void ConcreteAPI::destroyGenericStream(void* queue) {
  this->currentQueueBuffer().deleteQueue(queue);
}

void ConcreteAPI::syncStreamWithHost(void* streamPtr) {
  auto* queuePtr = static_cast<sycl::queue*>(streamPtr);
  this->currentQueueBuffer().syncQueueWithHost(queuePtr);
}

bool ConcreteAPI::isStreamWorkDone(void* streamPtr) {
  auto* queuePtr = static_cast<sycl::queue*>(streamPtr);

  // if we have an extension to query for an empty queue, only check that here
  // otherwise, synchronize
  // (AdaptiveCpp's get_wait_list does not help here: for an in-order queue that it does not
  // emulate, it always returns a newly submitted barrier, i.e. never an empty list)
#ifdef SYCL_EXT_ONEAPI_QUEUE_EMPTY
  return queuePtr->ext_oneapi_empty();
#elif defined(SYCL_KHR_QUEUE_EMPTY_QUERY)
  return queuePtr->khr_empty();
#else
  this->currentQueueBuffer().syncQueueWithHost(queuePtr);
  return true;
#endif
}

void ConcreteAPI::streamHostFunction(void* streamPtr, const std::function<void()>& function) {
  auto* queuePtr = static_cast<sycl::queue*>(streamPtr);

#ifdef __ACPP__
  // AdaptiveCpp has no host_task, and it evaluates a custom operation when it is submitted, not
  // once the operations before it on the queue have completed (cf. its
  // doc/enqueue-custom-operation.md). Hence, wait for them here and run the function right away.
  this->currentQueueBuffer().syncQueueWithHost(queuePtr);
  function();
#else
  queuePtr->submit([&](sycl::handler& h) { h.host_task([=]() { function(); }); });
#endif
}

void ConcreteAPI::streamWaitMemory(void* streamPtr, uint32_t* location, uint32_t value) {
  // for now, spin wait here
  auto* queuePtr = static_cast<sycl::queue*>(streamPtr);
  volatile uint32_t* spinLocation = location;
  queuePtr->single_task([=]() {
    while (true) {
      if (*spinLocation == value) {
        return;
      }

      // not yet supported by ACPP
#ifndef __ACPP__
      sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
#endif
    }
  });
}
