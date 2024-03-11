#include "DeviceType.h"
#include "SyclWrappedAPI.h"
#include "utils/logger.h"
#include "Internals.h"

#include <iostream>
#include <string>

using namespace device;

void ConcreteAPI::initDevices() {

  if (this->deviceInitialized) {
    throw new std::invalid_argument("Cannot initialize the devices twice!");
  }

  const auto concurrencyLevel = getMaxConcurrencyLevel(4);
  const auto rank = getMpiRankFromEnv();
  logDebug(rank) << "Init SYCL API devices ...";
  for (auto const &platform : cl::sycl::platform::get_platforms()) {
    for (auto const &device : platform.get_devices()) {

      auto platName = device.get_platform().get_info<cl::sycl::info::platform::name>();
      auto devName = device.get_info<cl::sycl::info::device::name>();
      auto type = device.get_info<cl::sycl::info::device::device_type>();

      if (type == cl::sycl::info::device_type::gpu) {
        if (devName.find("Intel") != std::string::npos) {
          if (platName.find("Level-Zero") == std::string::npos) {
            logDebug(rank) << "Skipping non level-zero Intel GPU" << devName << "on platform" << platName;
            continue;
          }
        }
      }

      DeviceContext *context = new DeviceContext{device, concurrencyLevel};
      this->availableDevices.push_back(context);
    }
  }

  sort(this->availableDevices.begin(), this->availableDevices.end(), [&](DeviceContext *c1, DeviceContext *c2) {
    return compare(c1->queueBuffer.getDefaultQueue().get_device(), c2->queueBuffer.getDefaultQueue().get_device());
  });

  std::ostringstream s;
  s << "Sorted available devices: ";
  int index = 0;
  for (int i = 0; i < this->availableDevices.size(); ++i) {
    if (i > 0) {
      s << "; ";
    }
    s << "(" << index << ") " << this->getDeviceName(i);
  }
  logDebug(rank) << s.str().c_str();

  this->setDevice(this->currentDeviceId);
  this->deviceInitialized = true;
  logDebug(rank) << "Device initialization succeeded";
}

void ConcreteAPI::setDevice(int id) {

  if (id < 0 || id >= this->getNumDevices()) {
    throw std::out_of_range{"Device index out of range"};
  }

  this->currentDeviceId = id;
  auto *next = this->availableDevices[id];
  this->currentDeviceStack = &next->stack;
  this->currentStatistics = &next->statistics;
  this->currentQueueBuffer = &next->queueBuffer;
  this->currentDefaultQueue = &this->currentQueueBuffer->getDefaultQueue();
  this->currentMemoryToSizeMap = &next->memoryToSizeMap;

  const auto rank = getMpiRankFromEnv();
  logDebug(rank) << "Switched to device: " << this->getDeviceName(id) << " by index " << id;
}

void ConcreteAPI::initialize() {}

void ConcreteAPI::allocateStackMem() {
  const auto rank = getMpiRankFromEnv();
  logDebug(rank) << "Allocating stack memory for device" << this->getDeviceName(this->currentDeviceId);

  this->currentDeviceStack->initMemory();
}

void ConcreteAPI::finalize() {
  if (m_isFinalized) {
    const auto rank = getMpiRankFromEnv();
    logWarning(rank) << "SYCL API is already finalized.";
    return;
  }
  for (auto *device : this->availableDevices) {
    delete device;
  }
  this->availableDevices.clear();
  this->availableDevices.shrink_to_fit();

  this->graphs.clear();

  this->currentDeviceStack = nullptr;
  this->currentStatistics = nullptr;
  this->currentQueueBuffer = nullptr;
  this->currentDefaultQueue = nullptr;
  this->currentMemoryToSizeMap = nullptr;

  this->m_isFinalized = true;
  this->deviceInitialized = false;
}

int ConcreteAPI::getNumDevices() { return this->availableDevices.size(); }

int ConcreteAPI::getDeviceId() {
  if (!deviceInitialized) {
    logError() << "Device has not been selected. Please, select device before requesting device Id";
  }
  return currentDeviceId;
}


size_t ConcreteAPI::getLaneSize() {
  return static_cast<size_t>(::internals::WARP_SIZE);
}

unsigned int ConcreteAPI::getMaxThreadBlockSize() {
  auto device = this->currentDefaultQueue->get_device();
  return device.get_info<cl::sycl::info::device::max_work_group_size>();
}

unsigned int ConcreteAPI::getMaxSharedMemSize() {
  auto device = this->currentDefaultQueue->get_device();
  return device.get_info<cl::sycl::info::device::local_mem_size>();
}

unsigned int ConcreteAPI::getGlobMemAlignment() {
  auto device = this->currentDefaultQueue->get_device();
  return 128; //ToDo: find attribute; not: device.get_info<info::device::mem_base_addr_align>();
}

void ConcreteAPI::syncDevice() { this->currentQueueBuffer->syncAllQueuesWithHost(); }

std::string ConcreteAPI::getDeviceInfoAsText(int id) {
  if (id < 0 || id >= this->getNumDevices())
    throw std::out_of_range{"Device index out of range"};

  auto device = this->availableDevices[id]->queueBuffer.getDefaultQueue().get_device();
  return this->getDeviceInfoAsText(device);
}
std::string ConcreteAPI::getCurrentDeviceInfoAsText() { return this->getDeviceInfoAsText(this->currentDeviceId); }

std::string ConcreteAPI::getDeviceInfoAsText(cl::sycl::device dev) {
  std::ostringstream info{};

  info << "platform:" << dev.get_platform().get_info<cl::sycl::info::platform::name>() << "\n";
  info << "    name:" << dev.get_info<cl::sycl::info::device::name>() << "\n";
  info << "    type: " << convertToString(dev.get_info<cl::sycl::info::device::device_type>()) << "\n";
  info << "    driver_version: " << dev.get_info<cl::sycl::info::device::driver_version>() << "\n";
  // if (dev.get_info<info::device::device_type>() != cl::sycl::info::device_type::host)
  // info << "    device id: " << dev.get() << "\n";

  return info.str();
}

bool isUnifiedMemoryDefault() {
  // suboptimal (i.e. we'd need to query if USM needs to be migrated or not), but there's probably nothing better for now
  return device.has<cl::sycl::aspect::usm_system_allocations>();
}

std::string ConcreteAPI::getApiName() {
  return "SYCL";
}

std::string ConcreteAPI::getDeviceName(int deviceId) {
  auto device = this->availableDevices[deviceId]->queueBuffer.getDefaultQueue().get_device();

  return device.get_info<cl::sycl::info::device::name>();
}

std::string ConcreteAPI::getPciAddress(int deviceId) {
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
  auto device = this->availableDevices[deviceId]->queueBuffer.getDefaultQueue().get_device();

  return device.get_info<sycl::ext::intel::info::device::pci_address>();
#else
  return "(unknown)";
#endif
}

void ConcreteAPI::putProfilingMark(const std::string &name, ProfilingColors color) {
  // ToDo: check if there is some similar functionality in VTUNE
}

void ConcreteAPI::popLastProfilingMark() {
  // ToDo: check if there is some similar functionality in VTUNE
}
