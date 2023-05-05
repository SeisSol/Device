#include "DeviceCircularQueueBuffer.h"

#include "utils/logger.h"

namespace device {
DeviceCircularQueueBuffer::DeviceCircularQueueBuffer(
    cl::sycl::device dev,
    std::function<void(cl::sycl::exception_list)> handler,
    size_t capacity)
    : queues{std::vector<cl::sycl::queue>(capacity)} {
  if (capacity <= 0)
    throw std::invalid_argument("Capacity must be at least 1!");

  this->defaultQueue = cl::sycl::queue{dev, handler, cl::sycl::property::queue::in_order()};
  this->genericQueue = cl::sycl::queue{dev, handler, cl::sycl::property::queue::in_order()};
  for (size_t i = 0; i < capacity; i++) {
    this->queues[i] = cl::sycl::queue{dev, handler, cl::sycl::property::queue::in_order()};
  }
}

cl::sycl::queue& DeviceCircularQueueBuffer::getDefaultQueue() {
  return defaultQueue;
}

cl::sycl::queue& DeviceCircularQueueBuffer::getGenericQueue() {
  return genericQueue;
}

cl::sycl::queue& DeviceCircularQueueBuffer::getNextQueue() {
  (++this->counter) %= getCapacity();
  return (this->queues[this->counter]);
}

void DeviceCircularQueueBuffer::resetIndex() {
  this->counter = 0;
}

size_t DeviceCircularQueueBuffer::getCapacity() {
  return queues.size();
}

void DeviceCircularQueueBuffer::syncQueueWithHost(cl::sycl::queue *queuePtr) {
  if (!this->exists(queuePtr))
    throw std::invalid_argument("DEVICE::ERROR: passed stream does not belong to circular stream buffer");

  queuePtr->wait_and_throw();
}

void DeviceCircularQueueBuffer::syncAllQueuesWithHost() {
  defaultQueue.wait_and_throw();
  for (auto &q : this->queues) {
    q.wait_and_throw();
  }
}

bool DeviceCircularQueueBuffer::exists(cl::sycl::queue *queuePtr) {
  bool isDefaultQueue = queuePtr == (&defaultQueue);
  bool isGenericQueue = queuePtr == (&genericQueue);

  bool isReservedQueue{true};
  for (auto& reservedQueue : queues) {
    if (queuePtr != (&reservedQueue)) {
      isReservedQueue = false;
      break;
    }
  }

  return isDefaultQueue || isGenericQueue || isReservedQueue;
}

} // namespace device
