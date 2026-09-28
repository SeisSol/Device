// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "HostWrappedAPI.h"
#include "utils/logger.h"

#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

using namespace device;

namespace {
// the same alignment as the GPU backends use; it keeps data layouts identical across backends
constexpr unsigned Alignment = 128;

void checkDeviceId(int deviceId) {
  if (deviceId != 0) {
    logError() << "The host backend has exactly one device (id 0); requested was" << deviceId;
  }
}
} // namespace

void ConcreteAPI::setDevice(int deviceId) {
  checkDeviceId(deviceId);
  status[StatusID::DriverApiInitialized] = true;
  status[StatusID::DeviceSelected] = true;
}

int ConcreteAPI::getDeviceId() {
  if (!status[StatusID::DeviceSelected]) {
    logError() << "Device has not been selected. Please, select device before requesting device Id";
  }
  return 0;
}

int ConcreteAPI::getNumDevices() { return 1; }

unsigned ConcreteAPI::getGlobMemAlignment() { return Alignment; }

std::string ConcreteAPI::getDeviceInfoAsText(int deviceId) {
  checkDeviceId(deviceId);
  std::ostringstream info;
  info << "name: " << getDeviceName(deviceId) << '\n';
  info << "execution: synchronous, on the calling host thread\n";
  info << "hardwareConcurrency: " << std::thread::hardware_concurrency() << '\n';
  info << "totalGlobalMem: " << getMaxAvailableMem() << '\n';
  info << "globMemAlignment: " << getGlobMemAlignment() << '\n';
  return info.str();
}

void ConcreteAPI::syncDevice() { isFlagSet<DeviceSelected>(status); }

std::string ConcreteAPI::getApiName() { return "Host"; }

std::string ConcreteAPI::getDeviceName(int deviceId) {
  checkDeviceId(deviceId);
  return "Host";
}

std::string ConcreteAPI::getPciAddress(int deviceId) {
  checkDeviceId(deviceId);
  return "";
}

size_t ConcreteAPI::getMaxAvailableMem() {
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long pageSize = sysconf(_SC_PAGE_SIZE);
  if (pages <= 0 || pageSize <= 0) {
    return 0;
  }
  return static_cast<size_t>(pages) * static_cast<size_t>(pageSize);
}

bool ConcreteAPI::isUnifiedMemoryDefault() {
  // the host accesses all memory of this backend directly
  return true;
}

void ConcreteAPI::initialize() {
  if (!status[StatusID::DeviceSelected]) {
    logError() << "Device has not been selected. Please, select device before calling initialize";
  }
  status[StatusID::InterfaceInitialized] = true;
}

void ConcreteAPI::finalize() {
  if (status[StatusID::InterfaceInitialized]) {
    if (!genericStreams.empty()) {
      logInfo() << "DEVICE::WARNING:" << genericStreams.size()
                << "device generic stream(s) were not deleted.";
      genericStreams.clear();
    }
    status[StatusID::InterfaceInitialized] = false;
  }
  m_isFinalized = true;
}

void ConcreteAPI::profilingMessage(const std::string& /*message*/) {}

void ConcreteAPI::putProfilingMark(const std::string& /*name*/, ProfilingColors /*color*/) {}

void ConcreteAPI::popLastProfilingMark() {}

void ConcreteAPI::setupPrinting(int /*rank*/) {}
