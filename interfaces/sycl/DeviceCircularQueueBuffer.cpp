// SPDX-FileCopyrightText: 2022 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DeviceCircularQueueBuffer.h"

#include "Internals.h"
#include "SyclWrappedAPI.h"
#include "utils/logger.h"

#include <sycl/sycl.hpp>

using namespace device::internals;

namespace device {

// very inconvenient, but AdaptiveCpp doesn't allow much freedom when constructing a property_list
#if defined(DEVICE_USE_GRAPH_CAPTURING) && defined(SYCL_EXT_INTEL_QUEUE_IMMEDIATE_COMMAND_LIST)
#define BASE_QUEUE_PROPERTIES                                                                      \
  sycl::property::queue::in_order{}, sycl::ext::intel::property::queue::no_immediate_command_list {}
#else
#define BASE_QUEUE_PROPERTIES sycl::property::queue::in_order()
#endif

QueueWrapper::QueueWrapper(const sycl::device& dev,
                           const std::function<void(sycl::exception_list l)>& handler)
    : queue{dev, handler, sycl::property_list{BASE_QUEUE_PROPERTIES}} {}

void QueueWrapper::synchronize() { waitCheck(queue); }
void QueueWrapper::dependency(QueueWrapper& other) {
  // improvising... Adding an empty event here, mimicking a CUDA-like event dependency
#if defined(HIPSYCL_EXT_QUEUE_WAIT_LIST) || defined(ACPP_EXT_QUEUE_WAIT_LIST) ||                   \
    defined(SYCL_EXT_ACPP_QUEUE_WAIT_LIST)
  auto waitList1 = other.queue.get_wait_list();
  auto waitList2 = queue.get_wait_list();
  queue.submit([&](sycl::handler& h) {
    h.depends_on(waitList1);
    h.depends_on(waitList2);
    DEVICE_SYCL_EMPTY_OPERATION(h);
  });
#else
  auto queueEvent = other.queue.submit([&](sycl::handler& h) { DEVICE_SYCL_EMPTY_OPERATION(h); });
  queue.submit([&](sycl::handler& h) { DEVICE_SYCL_EMPTY_OPERATION_WITH_EVENT(h, queueEvent); });
#endif
}

DeviceCircularQueueBuffer::DeviceCircularQueueBuffer(
    const sycl::device& dev,
    const std::function<void(sycl::exception_list)>& handler,
    size_t capacity)
    : queues{std::vector<QueueWrapper>(capacity)}, deviceReference(dev), handlerReference(handler) {
  if (capacity <= 0)
    throw std::invalid_argument("Capacity must be at least 1!");

  this->defaultQueue = QueueWrapper(dev, handler);
  this->genericQueue = QueueWrapper(dev, handler);
  for (size_t i = 0; i < capacity; i++) {
    this->queues[i] = QueueWrapper(dev, handler);
  }
}

sycl::queue& DeviceCircularQueueBuffer::getDefaultQueue() { return defaultQueue.queue; }

sycl::queue& DeviceCircularQueueBuffer::getGenericQueue() { return genericQueue.queue; }

sycl::queue& DeviceCircularQueueBuffer::getNextQueue() {
  (++this->counter) %= getCapacity();
  return (this->queues[this->counter].queue);
}

std::vector<sycl::queue> DeviceCircularQueueBuffer::allQueues() {
  std::vector<sycl::queue> queueCopy(queues.size() + 1);
  queueCopy[0] = defaultQueue.queue;
  for (size_t i = 0; i < queues.size(); ++i) {
    queueCopy[i + 1] = queues[i].queue;
  }
  return queueCopy;
}

sycl::queue* DeviceCircularQueueBuffer::newQueue(double priority) {
  // missing for ACPP: how can we even find out the allowed priority range conveniently now? :/

#ifdef SYCL_EXT_ONEAPI_QUEUE_PRIORITY
  const auto propertylist = [&]() -> sycl::property_list {
    if (priority <= 0.33) {
      return {BASE_QUEUE_PROPERTIES, sycl::ext::oneapi::property::queue::priority_low()};
    }
    if (priority >= 0.67) {
      return {BASE_QUEUE_PROPERTIES, sycl::ext::oneapi::property::queue::priority_high()};
    }
    return {BASE_QUEUE_PROPERTIES, sycl::ext::oneapi::property::queue::priority_normal()};
  }();
#else
  const auto propertylist{BASE_QUEUE_PROPERTIES};
#endif

  auto* queue = new sycl::queue{deviceReference, handlerReference, propertylist};
  externalQueues.emplace_back(queue);
  return queue;
}

void DeviceCircularQueueBuffer::deleteQueue(void* queue) {
  auto* queuePtr = static_cast<sycl::queue*>(queue);
  delete queuePtr;
}

void DeviceCircularQueueBuffer::resetIndex() { this->counter = 0; }

size_t DeviceCircularQueueBuffer::getCapacity() { return queues.size(); }

void DeviceCircularQueueBuffer::forkQueueDepencency() {
  for (auto& queue : this->queues) {
    queue.dependency(defaultQueue);
  }
}

void DeviceCircularQueueBuffer::joinQueueDepencency() {
  for (auto& queue : this->queues) {
    defaultQueue.dependency(queue);
  }
}

void DeviceCircularQueueBuffer::syncQueueWithHost(sycl::queue* queuePtr) { waitCheck(*queuePtr); }

void DeviceCircularQueueBuffer::syncAllQueuesWithHost() {
  defaultQueue.synchronize();
  for (auto& q : this->queues) {
    q.synchronize();
  }
  for (auto* q : this->externalQueues) {
    waitCheck(*q);
  }
}

bool DeviceCircularQueueBuffer::exists(sycl::queue* queuePtr) {
  bool isDefaultQueue = queuePtr == (&defaultQueue.queue);
  bool isGenericQueue = queuePtr == (&genericQueue.queue);

  bool isReservedQueue{true};
  for (auto& reservedQueue : queues) {
    if (queuePtr != (&reservedQueue.queue)) {
      isReservedQueue = false;
      break;
    }
  }

  bool isExternalQueue = false;
  for (auto& queue : externalQueues) {
    if (queuePtr == queue) {
      isExternalQueue = true;
      break;
    }
  }

  return isDefaultQueue || isGenericQueue || isReservedQueue || isExternalQueue;
}

} // namespace device
