#ifndef DEVICE_STATISTICS_H
#define DEVICE_STATISTICS_H

namespace device {
  struct Statistics {
    size_t allocatedMemBytes = 0;
    size_t allocatedUnifiedMemBytes = 0;
    size_t deallocatedMemBytes = 0;

    size_t explicitlyTransferredDataToDeviceBytes = 0;
    size_t explicitlyTransferredDataToHostBytes = 0;
  };
}

#endif //DEVICE_STATISTICS_H