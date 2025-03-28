// SPDX-FileCopyrightText: 2022-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DeviceCircularQueueBuffer.h"

#include "SyclWrappedAPI.h"
#include "utils/logger.h"

namespace device {

// very inconvenient, but AdaptiveCpp doesn't allow much freedom when constructing a property_list
#if defined(DEVICE_USE_GRAPH_CAPTURING) && defined(SYCL_EXT_INTEL_QUEUE_IMMEDIATE_COMMAND_LIST)
#define BASE_QUEUE_PROPERTIES cl::sycl::property::queue::in_order{}, cl::sycl::ext::intel::property::queue::no_immediate_command_list{}
#else
#define BASE_QUEUE_PROPERTIES cl::sycl::property::queue::in_order()
#endif

QueueWrapper::QueueWrapper(const cl::sycl::device& dev, const std::function<void(cl::sycl::exception_list l)>& handler)
: queue{dev, handler, cl::sycl::property_list{BASE_QUEUE_PROPERTIES}} {
}

void QueueWrapper::synchronize() {
  queue.wait_and_throw();
}
void QueueWrapper::dependency(QueueWrapper& other) {
  // improvising... Adding an empty event here, mimicking a CUDA-like event dependency
#if defined(HIPSYCL_EXT_QUEUE_WAIT_LIST) || defined(ACPP_EXT_QUEUE_WAIT_LIST) || defined(SYCL_EXT_ACPP_QUEUE_WAIT_LIST)
  auto waitList1 = other.queue.get_wait_list();
  auto waitList2 = queue.get_wait_list();
  queue.submit([&](sycl::handler& h) {
    h.depends_on(waitList1);
    h.depends_on(waitList2);
    DEVICE_SYCL_EMPTY_OPERATION(h);
  });
#else
  auto queueEvent = other.queue.submit([&](cl::sycl::handler& h) {
    DEVICE_SYCL_EMPTY_OPERATION(h);
  });
  queue.submit([&](cl::sycl::handler& h) {
    h.depends_on(queueEvent);
    DEVICE_SYCL_EMPTY_OPERATION(h);
  });
#endif
}

DeviceCircularQueueBuffer::DeviceCircularQueueBuffer(
    const cl::sycl::device& dev,
    const std::function<void(cl::sycl::exception_list)>& handler,
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

cl::sycl::queue& DeviceCircularQueueBuffer::getDefaultQueue() {
  return defaultQueue.queue;
}

cl::sycl::queue& DeviceCircularQueueBuffer::getGenericQueue() {
  return genericQueue.queue;
}

cl::sycl::queue& DeviceCircularQueueBuffer::getNextQueue() {
  (++this->counter) %= getCapacity();
  return (this->queues[this->counter].queue);
}

std::vector<cl::sycl::queue> DeviceCircularQueueBuffer::allQueues() {
  std::vector<cl::sycl::queue> queueCopy(queues.size() + 1);
  queueCopy[0] = defaultQueue.queue;
  for (size_t i = 0; i < queues.size(); ++i) {
    queueCopy[i+1] = queues[i].queue;
  }
  return queueCopy;
}

cl::sycl::queue* DeviceCircularQueueBuffer::newQueue(double priority) {
  // missing for ACPP: how can we even find out the allowed priority range conveniently now? :/

#ifdef SYCL_EXT_ONEAPI_QUEUE_PRIORITY
  const auto propertylist = [&]() -> cl::sycl::property_list {
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

  auto* queue = new cl::sycl::queue{deviceReference, handlerReference, propertylist};
  externalQueues.emplace_back(queue);
  return queue;
}

void DeviceCircularQueueBuffer::deleteQueue(void* queue) {
  auto *queuePtr = static_cast<cl::sycl::queue *>(queue);
  delete queuePtr;
}

void DeviceCircularQueueBuffer::resetIndex() {
  this->counter = 0;
}

size_t DeviceCircularQueueBuffer::getCapacity() {
  return queues.size();
}

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

void DeviceCircularQueueBuffer::syncQueueWithHost(cl::sycl::queue *queuePtr) {
  queuePtr->wait_and_throw();
}

void DeviceCircularQueueBuffer::syncAllQueuesWithHost() {
  defaultQueue.synchronize();
  for (auto &q : this->queues) {
    q.synchronize();
  }
  for (auto* q : this->externalQueues) {
    q->wait_and_throw();
  }
}

bool DeviceCircularQueueBuffer::exists(cl::sycl::queue *queuePtr) {
  bool isDefaultQueue = queuePtr == (&defaultQueue.queue);
  bool isGenericQueue = queuePtr == (&genericQueue.queue);

  bool isReservedQueue{true};
  for (auto& reservedQueue : queues) {
    if (queuePtr != (&reservedQueue.queue)) {
      isReservedQueue = false;
      break;
    }
  }

  return isDefaultQueue || isGenericQueue || isReservedQueue;
}

} // namespace device

