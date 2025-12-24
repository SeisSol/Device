// SPDX-FileCopyrightText: 2021 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_SYCL_DEVICECIRCULARQUEUEBUFFER_H_
#define SEISSOLDEVICE_INTERFACES_SYCL_DEVICECIRCULARQUEUEBUFFER_H_

#include <functional>
#include <stack>
#include <string.h>
#include <sycl/sycl.hpp>
#include <vector>

namespace device {
struct QueueWrapper {
  sycl::queue queue;
  QueueWrapper() = default;
  QueueWrapper(const sycl::device& dev, const std::function<void(sycl::exception_list l)>& f);

  void synchronize();
  void dependency(QueueWrapper& other);
};

class DeviceCircularQueueBuffer {
  public:
  /*
   * Creates a new circular buffer containing sycl queues. The buffer needs
   * an async exception handler and a capacity that is currently per default 8.
   */
  DeviceCircularQueueBuffer(const sycl::device& dev,
                            const std::function<void(sycl::exception_list l)>& f,
                            size_t capacity = 6);

  /*
   * Returns the default queue created by a device.
   */
  sycl::queue& getDefaultQueue();

  /*
   * Returns the generic queue created by a device.
   * Note, this queue can be used for asynchronous
   * memory copies
   */
  sycl::queue& getGenericQueue();

  /*
   * Returns the next queue within the capacity of this buffer.
   */
  sycl::queue& getNextQueue();

  sycl::queue* newQueue(double priority);

  void deleteQueue(void* queue);

  std::vector<sycl::queue> allQueues();

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
  void syncQueueWithHost(sycl::queue* queuePtr);

  /*
   * Synchronizes all queues from the buffer with the host device.
   */
  void syncAllQueuesWithHost();

  /*
   *Returns true if the queue pointer is available on the buffer.
   */
  bool exists(sycl::queue* queuePtr);

  void forkQueueDepencency();

  void joinQueueDepencency();

  private:
  QueueWrapper defaultQueue;
  QueueWrapper genericQueue;
  std::vector<QueueWrapper> queues;
  std::vector<sycl::queue*> externalQueues;
  size_t counter;
  sycl::device deviceReference;
  std::function<void(sycl::exception_list)> handlerReference;
};

} // namespace device

#endif // SEISSOLDEVICE_INTERFACES_SYCL_DEVICECIRCULARQUEUEBUFFER_H_
