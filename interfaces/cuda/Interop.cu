// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "AbstractAPI.h"
#include "CudaWrappedAPI.h"
#include "Internals.h"
#include "utils/logger.h"

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
  APIWRAP(cudaDeviceCanAccessPeer(&result, getDeviceId(), otherDeviceId));
  return result != 0;
}

void ConcreteAPI::switchPeerAccess(int otherDeviceId, bool enable) {
  if (enable) {
    APIWRAP(cudaDeviceEnablePeerAccess(otherDeviceId, 0));
  } else {
    APIWRAP(cudaDeviceDisablePeerAccess(otherDeviceId));
  }
}

std::vector<uint8_t> ConcreteAPI::makeIpcMemHandle(void* ptr) {
  cudaIpcMemHandle_t handle{};
  APIWRAP(cudaIpcGetMemHandle(&handle, ptr));
  return serialize(handle);
}

void* ConcreteAPI::openIpcMemHandle(const std::vector<uint8_t>& handle) {
  void* ptr = nullptr;
  APIWRAP(cudaIpcOpenMemHandle(
      &ptr, deserialize<cudaIpcMemHandle_t>(handle), cudaIpcMemLazyEnablePeerAccess));
  return ptr;
}

void ConcreteAPI::closeIpcMemHandle(void* ptr) { APIWRAP(cudaIpcCloseMemHandle(ptr)); }

std::vector<uint8_t> ConcreteAPI::makeIpcEventHandle(void* event) {
  cudaIpcEventHandle_t handle{};
  APIWRAP(cudaIpcGetEventHandle(&handle, static_cast<cudaEvent_t>(event)));
  return serialize(handle);
}

void* ConcreteAPI::openIpcEventHandle(const std::vector<uint8_t>& handle) {
  cudaEvent_t event{};
  APIWRAP(cudaIpcOpenEventHandle(&event, deserialize<cudaIpcEventHandle_t>(handle)));
  return static_cast<void*>(event);
}
