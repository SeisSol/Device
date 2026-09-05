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

  // if we have the oneAPI extension available, only check for an empty queue here
  // otherwise, synchronize
#ifdef SYCL_EXT_ONEAPI_QUEUE_EMPTY
  return queuePtr->ext_oneapi_empty();
#elif defined(HIPSYCL_EXT_QUEUE_WAIT_LIST) || defined(ACPP_EXT_QUEUE_WAIT_LIST) ||                 \
    defined(SYCL_EXT_ACPP_QUEUE_WAIT_LIST)
  // The wait list holds the events a newly submitted operation would have to depend on. Those
  // entries are not dropped once they have been reached, so an empty list means "nothing was
  // ever submitted", not "nothing is outstanding". Ask the events themselves instead.
  const auto waitList = queuePtr->get_wait_list();
  return std::all_of(waitList.begin(), waitList.end(), [](const sycl::event& event) {
    return event.get_info<sycl::info::event::command_execution_status>() ==
           sycl::info::event_command_status::complete;
  });
#else
  this->currentQueueBuffer().syncQueueWithHost(queuePtr);
  return true;
#endif
}

void ConcreteAPI::streamHostFunction(void* streamPtr, const std::function<void()>& function) {
  auto* queuePtr = static_cast<sycl::queue*>(streamPtr);

#ifdef __ACPP__
  queuePtr->DEVICE_SYCL_DIRECT_OPERATION_NAME([=](...) { function(); });
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
      // ">=", like the wait-value operations of the other backends: a counter that is written
      // once per step is past the awaited value by the time this runs often enough
      if (*spinLocation >= value) {
        return;
      }

      // not yet supported by ACPP
#ifndef __ACPP__
      sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
#endif
    }
  });
}
