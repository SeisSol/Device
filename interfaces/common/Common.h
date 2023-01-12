#ifndef DEVICE_INTERFACE_STATUS_H
#define DEVICE_INTERFACE_STATUS_H

#include <array>
#include <cassert>

namespace device {
enum StatusID {
  DriverApiInitialized = 0,
  DeviceSelected,
  InterfaceInitialized,
  CircularStreamBufferInitialized,
  StackMemAllocated,
  Count
};

using StatusT = std::array<bool, StatusID::Count>;

template<StatusID ID>
void isFlagSet(const StatusT& status) { assert(status[ID]); };

template<typename T, typename U>
U align(T number, U alignment) {
  size_t alignmentFactor = (number + alignment - 1) / alignment;
  return alignmentFactor * alignment;
}
}

#endif