#include "DeviceCircularQueueBuffer.h"

#include "utils/logger.h"

namespace device {
DeviceCircularQueueBuffer::DeviceCircularQueueBuffer(cl::sycl::device dev, std::function<void(cl::sycl::exception_list)> handler,
                                                     size_t capacity)
    : queues{std::vector<cl::sycl::queue>(capacity)} {
  if (capacity <= 0)
    throw std::invalid_argument("Capacity must be at least 1!");

  for (size_t i = 0; i < capacity; i++) {
    this->queues[i] = cl::sycl::queue{dev, handler, cl::sycl::property::queue::in_order()};
  }
}

cl::sycl::queue &DeviceCircularQueueBuffer::getDefaultQueue() { return this->queues[0]; }

cl::sycl::queue &DeviceCircularQueueBuffer::getNextQueue() {
  (++this->counter) %= getCapacity();
  return (this->queues[this->counter]);
}

void DeviceCircularQueueBuffer::resetIndex() { this->counter = 0; }

size_t DeviceCircularQueueBuffer::getCapacity() { return queues.size(); }

void DeviceCircularQueueBuffer::syncQueueWithHost(cl::sycl::queue *queuePtr) {
  if (!this->exists(queuePtr))
    throw std::invalid_argument("DEVICE::ERROR: passed stream does not belong to circular stream buffer");

  queuePtr->wait_and_throw();
}

void DeviceCircularQueueBuffer::syncAllQueuesWithHost() {
  for (auto &q : this->queues) {
    q.wait_and_throw();
  }
}

void DeviceCircularQueueBuffer::fastSync() { this->syncAllQueuesWithHost(); }

bool DeviceCircularQueueBuffer::exists(cl::sycl::queue *queuePtr) {
  auto itr = std::find(this->queues.begin(), this->queues.end(), *queuePtr);
  if (itr == this->queues.end()) {
    return false;
  }

  return true;
}

} // namespace device
