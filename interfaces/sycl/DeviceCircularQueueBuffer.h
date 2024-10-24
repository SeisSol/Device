#ifndef DEVICE_CIRCULAR_QUEUE_BUFFER_H
#define DEVICE_CIRCULAR_QUEUE_BUFFER_H

#include <CL/sycl.hpp>
#include <functional>
#include <stack>
#include <string.h>


namespace device {
struct QueueWrapper {
  cl::sycl::queue queue;
  QueueWrapper() = default;
  QueueWrapper(const cl::sycl::device& dev, const std::function<void(cl::sycl::exception_list l)>& f);

  void synchronize();
  void dependency(QueueWrapper& other);
};

class DeviceCircularQueueBuffer {
public:

  /*
   * Creates a new circular buffer containing sycl queues. The buffer needs
   * an async exception handler and a capacity that is currently per default 8.
   */
  DeviceCircularQueueBuffer(const cl::sycl::device& dev,
                            const std::function<void(cl::sycl::exception_list l)>& f,
                            size_t capacity = 6);

  /*
   * Returns the default queue created by a device.
   */
  cl::sycl::queue &getDefaultQueue();

  /*
   * Returns the generic queue created by a device.
   * Note, this queue can be used for asynchronous
   * memory copies
   */
  cl::sycl::queue &getGenericQueue();

  /*
   * Returns the next queue within the capacity of this buffer.
   */
  cl::sycl::queue &getNextQueue();

  cl::sycl::queue* newQueue(double priority);

  void deleteQueue(void* queue);

  std::vector<cl::sycl::queue> allQueues();

  /*
   * Resets the index to the current element.
   */
  void resetIndex();

  /*
   * Returns the default queue created by a device.
   */
  size_t getCapacity();

  /*
   * Synchronizes a queue from the buffer with the host device.
   */
  void syncQueueWithHost(cl::sycl::queue *queuePtr);

  /*
   * Synchronizes all queues from the buffer with the host device.
   */
  void syncAllQueuesWithHost();

  /*
   *Returns true if the queue pointer is available on the buffer.
   */
  bool exists(cl::sycl::queue *queuePtr);

  void forkQueueDepencency();

  void joinQueueDepencency();

private:
  QueueWrapper defaultQueue;
  QueueWrapper genericQueue;
  std::vector<QueueWrapper> queues;
  std::vector<QueueWrapper*> externalQueues;
  size_t counter;
  cl::sycl::device deviceReference;
  std::function<void(cl::sycl::exception_list)> handlerReference;
};

} // namespace device

#endif
