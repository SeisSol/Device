// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Common.h"

#include "utils/logger.h"

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_set>

namespace device::ownqueues {

double steadySeconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void check(hsa_status_t status, const char* call) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* text = nullptr;
    hsa_status_string(status, &text);
    logError() << "Own queues: HSA failed in" << call << "with" << (text != nullptr ? text : "?");
  }
}

void checkComgr(amd_comgr_status_t status, const char* call) {
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    logError() << "Own queues: comgr failed in" << call << "with" << static_cast<int>(status);
  }
}

void warnOnce(const std::string& message) {
  static std::mutex mutex;
  static std::unordered_set<std::string> warned;
  const std::lock_guard lock(mutex);
  if (warned.insert(message).second) {
    logWarning() << "Own queues:" << message;
  }
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

} // namespace device::ownqueues
