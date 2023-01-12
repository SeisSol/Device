#ifndef DEVICE_DATA_TYPES_H
#define DEVICE_DATA_TYPES_H

#include <cstddef>
#include <limits>

#if REAL_SIZE == 8
using real = double;
#elif REAL_SIZE == 4
using real = float;
#else
#error REAL_SIZE not supported.
#endif

namespace device {
struct DeviceGraphHandle {
  static const size_t invalidId{std::numeric_limits<size_t>::max()};
public:
  explicit DeviceGraphHandle() : graphId(invalidId) {}
  explicit DeviceGraphHandle(size_t id) : graphId(id) {}

  DeviceGraphHandle(const DeviceGraphHandle& other) = default;
  DeviceGraphHandle& operator=(const DeviceGraphHandle& other) = default;

  bool isInitialized() const {
    return graphId != invalidId;
  }

  operator bool() const {
    return isInitialized();
  }

  bool operator!() const {
    return !isInitialized();
  }

  size_t getGraphId() { return graphId; }

private:
  size_t graphId{invalidId};
};
} // namespace device
#endif // DEVICE_DATA_TYPES_H