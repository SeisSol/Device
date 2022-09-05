#include "DeviceType.h"
#include "SyclWrappedAPI.h"
#include "utils/logger.h"
#include "Internals.h"

#include <iostream>
#include <string>

using namespace device;
using namespace cl::sycl;
using namespace std;

void ConcreteAPI::initDevices() {

  if (this->deviceInitialized)
    throw new std::invalid_argument("can not init devices twice!");

  logInfo() << "init SYCL API devices ...";
  for (auto const &platform : platform::get_platforms()) {
    for (auto const &device : platform.get_devices()) {

      auto platName = device.get_platform().get_info<info::platform::name>();
      auto devName = device.get_info<info::device::name>();
      auto type = device.get_info<info::device::device_type>();

      if (type == cl::sycl::info::device_type::gpu) {
        if (devName.find("Intel") != std::string::npos) {
          if (platName.find("Level-Zero") == std::string::npos) {
            logDebug() << "skip non level-zero Intel GPU " << devName << " on platform " << platName;
            continue;
          }
        }
      }

      DeviceContext *context = new DeviceContext{device};
      this->availableDevices.push_back(context);
    }
  }

  sort(this->availableDevices.begin(), this->availableDevices.end(), [&](DeviceContext *c1, DeviceContext *c2) {
    return compare(c1->queueBuffer.getDefaultQueue().get_device(), c2->queueBuffer.getDefaultQueue().get_device());
  });

  stringstream s;
  s << "Sorted available devices: ";
  for (auto &dev : this->availableDevices) {
    s << this->getDeviceInfoAsText(dev->queueBuffer.getDefaultQueue().get_device());
  }
  logInfo() << s.str();

  this->setDevice(this->currentDeviceId);
  this->deviceInitialized = true;
  logInfo() << "device init succeeded";
}

void ConcreteAPI::setDevice(int id) {

  if (id < 0 || id >= this->getNumDevices())
    throw out_of_range{"device index out of range"};

  this->currentDeviceId = id;
  auto *next = this->availableDevices[id];
  this->currentDeviceStack = &next->stack;
  this->currentStatistics = &next->statistics;
  this->currentQueueBuffer = &next->queueBuffer;
  this->currentDefaultQueue = &this->currentQueueBuffer->getDefaultQueue();
  this->currentMemoryToSizeMap = &next->memoryToSizeMap;

  logDebug() << "switched to device: " << this->getCurrentDeviceInfoAsText() << " by index " << id;
}

void ConcreteAPI::initialize() {}

void ConcreteAPI::allocateStackMem() {
  logInfo() << "allocating stack memory for device: \n"
            << this->getCurrentDeviceInfoAsText()
            << "if DEVICE_STACK_MEM_SIZE is not set, the specified default value is used\n";

  this->currentDeviceStack->initMemory();
}

void ConcreteAPI::finalize() {
  if (m_isFinalized) {
    logWarning() << "SYCL API is already finalized!";
    return;
  }
  for (auto *device : this->availableDevices) {
    delete device;
  }
  this->availableDevices.clear();
  this->availableDevices.shrink_to_fit();

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
  return device.get_info<info::device::max_work_group_size>();
}

unsigned int ConcreteAPI::getMaxSharedMemSize() {
  auto device = this->currentDefaultQueue->get_device();
  return device.get_info<info::device::local_mem_size>();
}

unsigned int ConcreteAPI::getGlobMemAlignment() {
  auto device = this->currentDefaultQueue->get_device();
  return 128; //ToDo: find attribute; not: device.get_info<info::device::mem_base_addr_align>();
}

void ConcreteAPI::synchDevice() { this->currentQueueBuffer->syncAllQueuesWithHost(); }

string ConcreteAPI::getDeviceInfoAsText(int id) {
  if (id < 0 || id >= this->getNumDevices())
    throw out_of_range{"device index out of range"};

  auto device = this->availableDevices[id]->queueBuffer.getDefaultQueue().get_device();
  return this->getDeviceInfoAsText(device);
}
std::string ConcreteAPI::getCurrentDeviceInfoAsText() { return this->getDeviceInfoAsText(this->currentDeviceId); }

std::string ConcreteAPI::getDeviceInfoAsText(cl::sycl::device dev) {
  ostringstream info{};

  info << "platform:" << dev.get_platform().get_info<info::platform::name>() << "\n";
  info << "    name:" << dev.get_info<info::device::name>() << "\n";
  info << "    type: " << convertToString(dev.get_info<info::device::device_type>()) << "\n";
  info << "    driver_version: " << dev.get_info<info::device::driver_version>() << "\n";
  // if (dev.get_info<info::device::device_type>() != cl::sycl::info::device_type::host)
  // info << "    device id: " << dev.get() << "\n";

  return info.str();
}

void ConcreteAPI::putProfilingMark(const string &name, ProfilingColors color) {
  // ToDo: check if there is some similar functionality in VTUNE
}

void ConcreteAPI::popLastProfilingMark() {
  // ToDo: check if there is some similar functionality in VTUNE
}
