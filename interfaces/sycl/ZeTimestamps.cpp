// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ZeTimestamps.h"

#include "utils/logger.h"

#include <cstdint>
#include <mutex>

#ifdef DEVICE_SYCL_LEVEL_ZERO
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <variant>
#endif

namespace device {

#ifdef DEVICE_SYCL_LEVEL_ZERO
namespace {
// the events of a pool, and their timestamps in host memory, get allocated together
constexpr std::uint32_t PoolSize = 256;

ze_event_handle_t zeEvent(void* event) { return static_cast<ze_event_handle_t>(event); }

bool succeeded(ze_result_t result, const char* call) {
  if (result != ZE_RESULT_SUCCESS) {
    logWarning() << "Level Zero:" << call << "failed with" << static_cast<int>(result);
    return false;
  }
  return true;
}

// the immediate command list behind the queue (of the calling thread; the runtime keeps one per
// thread); nullptr if the queue has none, e.g. with batched command lists for graphs
ze_command_list_handle_t immediateList(sycl::queue& queue) {
  if (queue.get_backend() != sycl::backend::ext_oneapi_level_zero) {
    return nullptr;
  }
  const auto native = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
  const auto* list = std::get_if<ze_command_list_handle_t>(&native);
  return list != nullptr ? *list : nullptr;
}
} // namespace
#endif

ZeTimestamps::~ZeTimestamps() {
#ifdef DEVICE_SYCL_LEVEL_ZERO
  for (const auto& stamp : all) {
    zeEventDestroy(zeEvent(stamp.event));
  }
  for (auto* pool : pools) {
    zeEventPoolDestroy(static_cast<ze_event_pool_handle_t>(pool));
  }
  for (auto* block : values) {
    sycl::free(block, syclContext.value());
  }
#endif
}

bool ZeTimestamps::possible([[maybe_unused]] sycl::queue& queue) {
#ifdef DEVICE_SYCL_LEVEL_ZERO
  return immediateList(queue) != nullptr;
#else
  return false;
#endif
}

bool ZeTimestamps::setUp([[maybe_unused]] sycl::queue& queue) {
#ifdef DEVICE_SYCL_LEVEL_ZERO
  if (tried) {
    return usable;
  }
  tried = true;
  const auto queueContext = queue.get_context();
  context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queueContext);
  device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
  ze_device_properties_t properties{};
  properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  if (!succeeded(zeDeviceGetProperties(static_cast<ze_device_handle_t>(device), &properties),
                 "zeDeviceGetProperties")) {
    return false;
  }
  // (in nanoseconds for this type of the structure; ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2 gives
  // cycles per second instead)
  nanosecondsPerTick = static_cast<double>(properties.timerResolution);
  validMask = properties.timestampValidBits >= 64
                  ? ~std::uint64_t{0}
                  : (std::uint64_t{1} << properties.timestampValidBits) - 1;
  syclContext = queueContext;
  usable = nanosecondsPerTick > 0 && properties.timestampValidBits > 1;
  if (usable) {
    logInfo() << "Device: the events with timing take the global timestamp of Level Zero, with"
              << nanosecondsPerTick << "ns per tick and" << properties.timestampValidBits
              << "valid bits.";
  }
  return usable;
#else
  return false;
#endif
}

bool ZeTimestamps::acquire([[maybe_unused]] Stamp& stamp) {
#ifdef DEVICE_SYCL_LEVEL_ZERO
  if (available.empty()) {
    ze_event_pool_desc_t poolDescription{};
    poolDescription.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    poolDescription.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    poolDescription.count = PoolSize;
    auto* zeDevice = static_cast<ze_device_handle_t>(device);
    ze_event_pool_handle_t pool = nullptr;
    if (!succeeded(
            zeEventPoolCreate(
                static_cast<ze_context_handle_t>(context), &poolDescription, 1, &zeDevice, &pool),
            "zeEventPoolCreate")) {
      return false;
    }
    auto* block = sycl::malloc_host<std::uint64_t>(PoolSize, syclContext.value());
    if (block == nullptr) {
      zeEventPoolDestroy(pool);
      return false;
    }
    pools.push_back(pool);
    values.push_back(block);
    for (std::uint32_t i = 0; i < PoolSize; ++i) {
      ze_event_desc_t eventDescription{};
      eventDescription.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
      eventDescription.index = i;
      eventDescription.signal = ZE_EVENT_SCOPE_FLAG_HOST;
      eventDescription.wait = ZE_EVENT_SCOPE_FLAG_HOST;
      ze_event_handle_t event = nullptr;
      if (!succeeded(zeEventCreate(pool, &eventDescription, &event), "zeEventCreate")) {
        return false;
      }
      all.push_back(Stamp{block + i, event, false});
      available.push_back(all.back());
    }
  }
  stamp = available.back();
  available.pop_back();
  return true;
#else
  return false;
#endif
}

bool ZeTimestamps::record([[maybe_unused]] sycl::queue& queue, [[maybe_unused]] Stamp& stamp) {
#ifdef DEVICE_SYCL_LEVEL_ZERO
  auto* list = immediateList(queue);
  if (list == nullptr) {
    return false;
  }
  {
    const std::scoped_lock lock(mutex);
    if (!setUp(queue) || (stamp.event == nullptr && !acquire(stamp))) {
      return false;
    }
  }
  auto* event = zeEvent(stamp.event);
  if (stamp.recorded) {
    // its previous timestamp has to be there before the memory gets written again
    zeEventHostSynchronize(event, UINT64_MAX);
  }
  zeEventHostReset(event);
  *stamp.value = 0;
  stamp.recorded = false;
  // the barrier first: the runtime keeps the order of an in-order queue by events, so its
  // immediate command list need not be in order by itself
  if (!succeeded(zeCommandListAppendBarrier(list, nullptr, 0, nullptr),
                 "zeCommandListAppendBarrier") ||
      !succeeded(zeCommandListAppendWriteGlobalTimestamp(list, stamp.value, event, 0, nullptr),
                 "zeCommandListAppendWriteGlobalTimestamp")) {
    return false;
  }
  stamp.recorded = true;
  return true;
#else
  return false;
#endif
}

bool ZeTimestamps::recordNow([[maybe_unused]] Stamp& stamp) {
#ifdef DEVICE_SYCL_LEVEL_ZERO
  {
    const std::scoped_lock lock(mutex);
    if (!usable || (stamp.event == nullptr && !acquire(stamp))) {
      return false;
    }
  }
  auto* event = zeEvent(stamp.event);
  if (stamp.recorded) {
    zeEventHostSynchronize(event, UINT64_MAX);
  }
  zeEventHostReset(event);
  stamp.recorded = false;
  std::uint64_t hostTime = 0;
  std::uint64_t deviceTime = 0;
  if (!succeeded(zeDeviceGetGlobalTimestamps(
                     static_cast<ze_device_handle_t>(device), &hostTime, &deviceTime),
                 "zeDeviceGetGlobalTimestamps")) {
    return false;
  }
  *stamp.value = deviceTime;
  zeEventHostSignal(event);
  stamp.recorded = true;
  return true;
#else
  return false;
#endif
}

bool ZeTimestamps::completed([[maybe_unused]] const Stamp& stamp) const {
#ifdef DEVICE_SYCL_LEVEL_ZERO
  return zeEventQueryStatus(zeEvent(stamp.event)) == ZE_RESULT_SUCCESS;
#else
  return true;
#endif
}

void ZeTimestamps::wait([[maybe_unused]] const Stamp& stamp) const {
#ifdef DEVICE_SYCL_LEVEL_ZERO
  zeEventHostSynchronize(zeEvent(stamp.event), UINT64_MAX);
#endif
}

double ZeTimestamps::seconds(const Stamp& start, const Stamp& end) const {
  // (the device writes the values; read them once their events say they are there)
  const std::uint64_t first = *static_cast<const volatile std::uint64_t*>(start.value);
  const std::uint64_t last = *static_cast<const volatile std::uint64_t*>(end.value);
  // the difference within the valid bits of the timestamps, as a signed number
  const std::uint64_t ticks = (last - first) & validMask;
  const std::uint64_t negative = (validMask >> 1) + 1;
  const double signedTicks =
      ticks >= negative ? -static_cast<double>(validMask - ticks + 1) : static_cast<double>(ticks);
  return signedTicks * nanosecondsPerTick * 1e-9;
}

void ZeTimestamps::release(Stamp& stamp) {
  if (stamp.event == nullptr) {
    return;
  }
  if (stamp.recorded) {
    // the device may still write it
    wait(stamp);
  }
  stamp.recorded = false;
  const std::scoped_lock lock(mutex);
  available.push_back(stamp);
  stamp = Stamp{};
}

} // namespace device
