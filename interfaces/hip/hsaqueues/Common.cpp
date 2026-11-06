// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Common.h"

#include "utils/logger.h"

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>

namespace device::hsaqueues {

double steadySeconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void check(hsa_status_t status, const char* call) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* text = nullptr;
    hsa_status_string(status, &text);
    logError() << "HSA queues: HSA failed in" << call << "with" << (text != nullptr ? text : "?");
  }
}

void checkComgr(amd_comgr_status_t status, const char* call) {
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    logError() << "HSA queues: comgr failed in" << call << "with" << static_cast<int>(status);
  }
}

void warnOnce(const std::string& message) {
  static std::mutex mutex;
  static std::unordered_set<std::string> warned;
  const std::lock_guard lock(mutex);
  if (warned.insert(message).second) {
    logWarning() << "HSA queues:" << message;
  }
}

std::optional<std::string> settingText(const std::string& name) {
  if (const char* value = std::getenv(("DEVICE_HSA_QUEUE_" + name).c_str()); value != nullptr) {
    return value;
  }
  if (const char* value = std::getenv(("DEVICE_OWN_QUEUE_" + name).c_str()); value != nullptr) {
    warnOnce("DEVICE_OWN_QUEUE_" + name + " is deprecated, use DEVICE_HSA_QUEUE_" + name);
    return value;
  }
  return std::nullopt;
}

/// the device can access the host memory: its address there, otherwise nullptr
const void* deviceAddress(const void* ptr) {
  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  if (hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS ||
      info.type == HSA_EXT_POINTER_TYPE_UNKNOWN) {
    return nullptr;
  }
  if (info.agentBaseAddress == nullptr || info.hostBaseAddress == nullptr ||
      info.agentBaseAddress == info.hostBaseAddress) {
    return ptr;
  }
  return static_cast<const char*>(info.agentBaseAddress) +
         (static_cast<const char*>(ptr) - static_cast<const char*>(info.hostBaseAddress));
}

} // namespace device::hsaqueues
