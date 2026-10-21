// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "AbstractAPI.h"
#include "HipWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"

#include <cstdint>
#include <vector>

using namespace device;

namespace {

template <typename T>
std::vector<uint8_t> serialize(const T& data) {
  std::vector<uint8_t> ser(sizeof(T));
  memcpy(ser.data(), &data, sizeof(T));
  return ser;
}

template <typename T>
T deserialize(const std::vector<uint8_t>& ser) {
  T data{};
  memcpy(&data, ser.data(), sizeof(T));
  return data;
}

} // namespace

bool ConcreteAPI::canAccessPeer(int otherDeviceId) {
  int result = 0;
  APIWRAP(hipDeviceCanAccessPeer(&result, getDeviceId(), otherDeviceId));
  return result != 0;
}

void ConcreteAPI::switchPeerAccess(int otherDeviceId, bool enable) {
  if (enable) {
    APIWRAP(hipDeviceEnablePeerAccess(otherDeviceId, 0));
  } else {
    APIWRAP(hipDeviceDisablePeerAccess(otherDeviceId));
  }
}

std::vector<uint8_t> ConcreteAPI::makeIpcMemHandle(void* ptr) {
  hipIpcMemHandle_t handle{};
  APIWRAP(hipIpcGetMemHandle(&handle, ptr));
  return serialize(handle);
}

void* ConcreteAPI::openIpcMemHandle(const std::vector<uint8_t>& handle) {
  void* ptr = nullptr;
  APIWRAP(hipIpcOpenMemHandle(
      &ptr, deserialize<hipIpcMemHandle_t>(handle), hipIpcMemLazyEnablePeerAccess));
  return ptr;
}

void ConcreteAPI::closeIpcMemHandle(void* ptr) { APIWRAP(hipIpcCloseMemHandle(ptr)); }

std::vector<uint8_t> ConcreteAPI::makeIpcEventHandle(void* event) {
  hipIpcEventHandle_t handle{};
  APIWRAP(hipIpcGetEventHandle(&handle, static_cast<hipEvent_t>(event)));
  return serialize(handle);
}

void* ConcreteAPI::openIpcEventHandle(const std::vector<uint8_t>& handle) {
  hipEvent_t event{};
  APIWRAP(hipIpcOpenEventHandle(&event, deserialize<hipIpcEventHandle_t>(handle)));
  return static_cast<void*>(event);
}
