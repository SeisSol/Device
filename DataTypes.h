#ifndef DEVICE_DATA_TYPES_H
#define DEVICE_DATA_TYPES_H

#include <limits>

#if REAL_SIZE == 8
using real = double;
#elif REAL_SIZE == 4
using real = float;
#else
#error REAL_SIZE not supported.
#endif

namespace device {
struct deviceGraphHandle {
  bool isInitialized() const {
    return graphID != std::numeric_limits<size_t>::max();
  }
  bool operator!() const {
    return !isInitialized();
  }
  size_t graphID{std::numeric_limits<size_t>::max()};
};

using deviceStreamPtr = void*;
}

#endif // DEVICE_DATA_TYPES_H