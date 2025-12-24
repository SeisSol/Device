// SPDX-FileCopyrightText: 2023 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_USMALLOCATOR_H_
#define SEISSOLDEVICE_USMALLOCATOR_H_

#include "device.h"

#include <cstdlib>
#include <stdexcept>

namespace device {
template <typename T>
class UsmAllocator {
  public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using void_pointer = void*;
  using const_void_pointer = const void*;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  UsmAllocator() noexcept = delete;
  UsmAllocator(device::DeviceInstance& instance) noexcept : api(instance.api) {}

  UsmAllocator(const UsmAllocator&) noexcept = default;
  UsmAllocator(UsmAllocator&&) noexcept = default;

  UsmAllocator& operator=(const UsmAllocator&) = delete;
  UsmAllocator& operator=(UsmAllocator&&) = default;

  template <typename U>
  UsmAllocator(const UsmAllocator<U>& other) noexcept : api(other.api) {}

  pointer allocate(size_type numElements) {
    auto* ptr = static_cast<pointer>(api->allocUnifiedMem(sizeof(T) * numElements));
    if (!ptr) {
      throw std::runtime_error("device::UsmAllocator: Allocation failed");
    }
    return ptr;
  }

  void deallocate(T* ptr, std::size_t) {
    if (ptr) {
      api->freeGlobMem(ptr);
    }
  }

  template <typename U>
  friend bool operator==(const UsmAllocator<T>& allocator, const UsmAllocator<U>& otherAllocator) {
    return allocator.api == otherAllocator.api;
  }

  template <typename U>
  friend bool operator!=(const UsmAllocator<T>& allocator, const UsmAllocator<U>& otherAllocator) {
    return !(allocator == otherAllocator);
  }

  private:
  template <typename U>
  friend class UsmAllocator;
  device::AbstractAPI* api{nullptr};
};
} // namespace device

#endif // SEISSOLDEVICE_USMALLOCATOR_H_
