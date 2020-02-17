#ifndef DEVICE_STATISTICS_H
#define DEVICE_STATISTICS_H

namespace device {
  struct Statistics {
    size_t AllocatedMemBytes = 0;
    size_t DeallocatedMemBytes = 0;

    size_t ExplicitlyTransferredDataToDeviceBytes = 0;
    size_t ExplicitlyTransferredDataToHostBytes = 0;
  };
}

#endif //DEVICE_STATISTICS_H