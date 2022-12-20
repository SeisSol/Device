#ifndef DEVICE_CIRCULAR_QUEUE_BUFFER_H
#define DEVICE_CIRCULAR_QUEUE_BUFFER_H

#include <CL/sycl.hpp>
#include <functional>
#include <stack>
#include <string.h>


namespace device {
class DeviceCircularQueueBuffer {
public:

  /*
   * Creates a new circular buffer containing sycl queues. The buffer needs
   * an async exception handler and a capacity that is currently per default 8.
   */
  DeviceCircularQueueBuffer(cl::sycl::device dev, std::function<void(cl::sycl::exception_list l)> f, size_t capacity = 7);

  /*
   * Returns the default queue created by a device.
   */
  cl::sycl::queue &getDefaultQueue();

  /*
   * Returns the next queue within the capacity of this buffer.
   */
  cl::sycl::queue &getNextQueue();

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

private:
  cl::sycl::queue defaultQueue;
  std::vector<cl::sycl::queue> queues;
  size_t counter;
};

} // namespace device

#endif
