#include "DeviceType.h"
#include "SyclWrappedAPI.h"
#include "utils/logger.h"

#include <iostream>
#include <string>

using namespace device;
using namespace cl::sycl;
using namespace std;

void ConcreteAPI::initialize() {
  if (initialized)
    throw std::invalid_argument("can not initialize SYCL twice");

  logInfo() << "init SYCL API wrapper ...";
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
    return compare(c1->queueBuffer.getDefaultQueue().get_device().get_info<cl::sycl::info::device::device_type>(),
                   c2->queueBuffer.getDefaultQueue().get_device().get_info<cl::sycl::info::device::device_type>());
  });

  stringstream s;
  s << "Sorted available devices: ";
  for (auto &dev : this->availableDevices) {
    s << this->getDeviceInfoAsText(dev->queueBuffer.getDefaultQueue().get_device());
  }
  logInfo() << s.str();

  this->setDevice(this->currentDeviceId);
  this->initialized = true;
  logInfo() << "init succeeded";
}

void ConcreteAPI::allocateStackMem() {
  logInfo() << "allocating stack memory for device: \n"
            << this->getCurrentDeviceInfoAsText()
            << "if DEVICE_STACK_MEM_SIZE is not set, the specified default value is used\n";

  this->currentDeviceStack->initMemory();
}

void ConcreteAPI::finalize() {
  if (m_isFinalized)
    throw std::invalid_argument("API is already finalized!");

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
  this->initialized = false;
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

int ConcreteAPI::getNumDevices() { return this->availableDevices.size(); }

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

void ConcreteAPI::synchDevice() { this->currentDefaultQueue->wait_and_throw(); }

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
  info << "    driver_version:" << dev.get_info<info::device::driver_version>() << "\n";

  return info.str();
}

void ConcreteAPI::putProfilingMark(const string &name, ProfilingColors color) {
  // ToDo: check if there is some similar functionality in VTUNE
}

void ConcreteAPI::popLastProfilingMark() {
  // ToDo: check if there is some similar functionality in VTUNE
}
