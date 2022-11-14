
#include "DeviceContext.h"
#include "DeviceType.h"
#include "utils/logger.h"

namespace device {
DeviceContext::DeviceContext(const cl::sycl::device &targetDevice)
    : queueBuffer{DeviceCircularQueueBuffer{targetDevice, [&](cl::sycl::exception_list l) { onExceptionOccurred(l); }}},
      stack{DeviceStack{queueBuffer.getDefaultQueue()}}, statistics{Statistics{}} {}

void DeviceContext::onExceptionOccurred(cl::sycl::exception_list &exceptions) {
  for (auto &excep : exceptions) {
    try {
      rethrow_exception(excep);
    } catch (const cl::sycl::exception &e) {
      logError() << e.what();
    }
  }
}
} // namespace device
