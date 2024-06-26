#include "DeviceCircularQueueBuffer.h"

#include "utils/logger.h"

namespace device {

#if defined(DEVICE_USE_GRAPH_CAPTURING) && defined(SYCL_EXT_INTEL_QUEUE_IMMEDIATE_COMMAND_LIST)
auto sycl_queue_properties() {
  return cl::sycl::property_list(cl::sycl::property::queue::in_order{},
                                 cl::sycl::ext::intel::property::queue::no_immediate_command_list{});
}
#else
auto sycl_queue_properties() { return cl::sycl::property_list(cl::sycl::property::queue::in_order{}); }
#endif

QueueWrapper::QueueWrapper(const cl::sycl::device& dev, const std::function<void(cl::sycl::exception_list l)>& handler)
: queue{dev, handler, sycl_queue_properties()} {
}

void QueueWrapper::synchronize() {
  queue.wait_and_throw();
}
void QueueWrapper::dependency(QueueWrapper& other) {
  // improvising... Adding an empty event here, mimicking a CUDA-like event dependency
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER) && !defined(DEVICE_USE_GRAPH_CAPTURING)
  auto queueEvent = other.queue.ext_oneapi_submit_barrier();
  queue.ext_oneapi_submit_barrier({queueEvent});
#else
  // mimick generically using a cheap dummy task
  auto queueEvent = other.queue.submit([&](cl::sycl::handler& h) {
    h.single_task([=](){});
  });
  queue.submit([&](cl::sycl::handler& h) {
    h.depends_on(queueEvent);
    h.single_task([=](){});
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

cl::sycl::queue DeviceCircularQueueBuffer::newQueue() {
  return cl::sycl::queue{deviceReference, handlerReference, sycl_queue_properties()};
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
  if (!this->exists(queuePtr))
    throw std::invalid_argument("DEVICE::ERROR: passed stream does not belong to circular stream buffer");

  queuePtr->wait_and_throw();
}

void DeviceCircularQueueBuffer::syncAllQueuesWithHost() {
  defaultQueue.synchronize();
  for (auto &q : this->queues) {
    q.synchronize();
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
