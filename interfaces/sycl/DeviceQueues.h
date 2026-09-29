// SPDX-FileCopyrightText: 2021 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_SYCL_DEVICEQUEUES_H_
#define SEISSOLDEVICE_INTERFACES_SYCL_DEVICEQUEUES_H_

#include <functional>
#include <mutex>
#include <sycl/sycl.hpp>
#include <vector>

namespace device {

/*
 * Owns the queues of one device: the default queue every caller shares, and the queues handed out
 * through newQueue.
 */
class DeviceQueues {
  public:
  DeviceQueues(const sycl::device& dev, const std::function<void(sycl::exception_list l)>& f);
  ~DeviceQueues();

  DeviceQueues(const DeviceQueues&) = delete;
  DeviceQueues& operator=(const DeviceQueues&) = delete;

  /*
   * Returns the default queue of this device.
   */
  sycl::queue& getDefaultQueue();

  /*
   * Creates a queue owned by this device. `priority` follows the convention of
   * AbstractAPI::createStream: 0 the lowest, 1 the highest, NaN the runtime default.
   */
  sycl::queue* newQueue(double priority);

  /*
   * Destroys a queue obtained from newQueue.
   */
  void deleteQueue(void* queue);

  /*
   * Synchronizes one queue with the host.
   */
  void syncQueueWithHost(sycl::queue* queuePtr);

  /*
   * Synchronizes every queue of this device with the host.
   */
  void syncAllQueuesWithHost();

  /*
   * Returns true if the queue belongs to this device.
   */
  bool exists(sycl::queue* queuePtr);

  private:
  sycl::queue defaultQueue;
  // guards externalQueues, which callers add to and remove from while other threads walk it
  std::mutex queueMutex;
  std::vector<sycl::queue*> externalQueues;
  sycl::device deviceReference;
  std::function<void(sycl::exception_list)> handlerReference;
};

} // namespace device

#endif // SEISSOLDEVICE_INTERFACES_SYCL_DEVICEQUEUES_H_
