#ifndef DEVICE_SUBMODULE_DEVICECONTEXT_H
#define DEVICE_SUBMODULE_DEVICECONTEXT_H

#include "DeviceCircularQueueBuffer.h"
#include "DeviceStack.h"
#include "Statistics.h"

namespace device {

/*
 * Container class holding the queue, stack, and buffer of the current device
 */
class DeviceContext {
public:
  DeviceContext(const cl::sycl::device &targetDevice);
  unordered_map<void *, size_t> memoryToSizeMap;
  DeviceCircularQueueBuffer queueBuffer;
  DeviceStack stack;
  Statistics statistics;
private:
  void onExceptionOccurred(exception_list &exceptions);
};
} // namespace device
#endif // DEVICE_SUBMODULE_DEVICECONTEXT_H
