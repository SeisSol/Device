// SPDX-FileCopyrightText: 2022 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DeviceQueues.h"

#include "Internals.h"
#include "SyclWrappedAPI.h"
#include "utils/logger.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <sycl/sycl.hpp>

using namespace device::internals;

namespace device {

// very inconvenient, but AdaptiveCpp doesn't allow much freedom when constructing a property_list
#if defined(DEVICE_USE_GRAPH_CAPTURING) && defined(SYCL_EXT_INTEL_QUEUE_IMMEDIATE_COMMAND_LIST)
#define IMMEDIATE_COMMAND_LIST_PROPERTY                                                            \
  , sycl::ext::intel::property::queue::no_immediate_command_list {}
#else
#define IMMEDIATE_COMMAND_LIST_PROPERTY
#endif

// profiling is what makes the timings of AbstractAPI::timespanEvents available, and it costs on
// every submission, so it follows the build option rather than being on by default
#ifdef PROFILING_ENABLED
#define PROFILING_PROPERTY                                                                         \
  , sycl::property::queue::enable_profiling {}
#else
#define PROFILING_PROPERTY
#endif

#define BASE_QUEUE_PROPERTIES                                                                      \
  sycl::property::queue::in_order {}                                                               \
  IMMEDIATE_COMMAND_LIST_PROPERTY PROFILING_PROPERTY

DeviceQueues::DeviceQueues(const sycl::device& dev,
                           const std::function<void(sycl::exception_list)>& handler)
    : defaultQueue{dev, handler, sycl::property_list{BASE_QUEUE_PROPERTIES}}, deviceReference(dev),
      handlerReference(handler) {}

DeviceQueues::~DeviceQueues() {
  if (!externalQueues.empty()) {
    logInfo() << "DEVICE::WARNING:" << externalQueues.size()
              << "device generic stream(s) were not deleted.";
  }
  for (auto* queue : externalQueues) {
    delete queue;
  }
}

sycl::queue& DeviceQueues::getDefaultQueue() { return defaultQueue; }

sycl::queue* DeviceQueues::newQueue(double priority) {
  // missing for ACPP: how can we even find out the allowed priority range conveniently now? :/

#ifdef SYCL_EXT_ONEAPI_QUEUE_PRIORITY
  const sycl::property_list propertylist = [&]() -> sycl::property_list {
    if (std::isnan(priority)) {
      return {BASE_QUEUE_PROPERTIES};
    }
    if (priority <= 0.33) {
      return {BASE_QUEUE_PROPERTIES, sycl::ext::oneapi::property::queue::priority_low()};
    }
    if (priority >= 0.67) {
      return {BASE_QUEUE_PROPERTIES, sycl::ext::oneapi::property::queue::priority_high()};
    }
    return {BASE_QUEUE_PROPERTIES, sycl::ext::oneapi::property::queue::priority_normal()};
  }();
#else
  const sycl::property_list propertylist{BASE_QUEUE_PROPERTIES};
#endif

  auto* queue = new sycl::queue{deviceReference, handlerReference, propertylist};

  const std::lock_guard<std::mutex> lock(queueMutex);
  externalQueues.emplace_back(queue);
  return queue;
}

void DeviceQueues::deleteQueue(void* queue) {
  auto* queuePtr = static_cast<sycl::queue*>(queue);
  const std::lock_guard<std::mutex> lock(queueMutex);

  // The queue has to leave the list before it is freed: syncAllQueuesWithHost walks that list,
  // so a stale entry turns into a use-after-free at the next device-wide synchronization, far
  // away from whoever destroyed the queue.
  const auto entry = std::find(externalQueues.begin(), externalQueues.end(), queuePtr);
  if (entry == externalQueues.end()) {
    logWarning() << "Tried to destroy a stream that this device does not know about. It has "
                    "either been destroyed already or belongs to a different device; not "
                    "freeing it again.";
    return;
  }

  externalQueues.erase(entry);
  delete queuePtr;
}

void DeviceQueues::syncQueueWithHost(sycl::queue* queuePtr) { waitCheck(*queuePtr); }

void DeviceQueues::syncAllQueuesWithHost() {
  waitCheck(defaultQueue);

  // copied under the lock: waiting on a queue takes as long as the work on it, and holding the
  // lock for that would block every thread that wants to create or destroy one
  const auto queues = [this]() {
    const std::lock_guard<std::mutex> lock(queueMutex);
    return externalQueues;
  }();

  for (auto* queue : queues) {
    waitCheck(*queue);
  }
}

bool DeviceQueues::exists(sycl::queue* queuePtr) {
  if (queuePtr == &defaultQueue) {
    return true;
  }

  const std::lock_guard<std::mutex> lock(queueMutex);
  return std::find(externalQueues.begin(), externalQueues.end(), queuePtr) != externalQueues.end();
}

} // namespace device
