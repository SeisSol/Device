#include "DummyInterface.h"
#include <iostream>
#include <string>

using namespace device;

void ConcreteInterface::setDevice(int DeviceId) {
  std::cout << "picking device " << DeviceId << " as a working one" << std::endl;
}


int ConcreteInterface::getNumDevices() {
  std::cout << "querying num. devices available" << std::endl;
  return 1;
}

std::string ConcreteInterface::getDeviceInfoAsText(int DeviceId) {
  return std::string("It is a dummy device");
}

void ConcreteInterface::synchDevice() {
  std::cout << "synchronizing the controlled device" << std::endl;
}

void* ConcreteInterface::allocGlobMem(size_t Size) {
  std::cout << "allocating glob mem. size of " << Size << " bytes" << std::endl;
  return nullptr;
}


void* ConcreteInterface::allocUnifiedMem(size_t Size) {
  std::cout << "allocating unified mem. size of " << Size << " bytes" << std::endl;
  return nullptr;
}


void* ConcreteInterface::allocPinnedMem(size_t Size) {
  std::cout << "allocating pinned mem. size of " << Size << " bytes" << std::endl;
  return nullptr;
}


void ConcreteInterface::freeMem(void *DevPtr) {
  std::cout << "free memory" << std::endl;
}


void ConcreteInterface::freePinnedMem(void *DevPtr) {
  std::cout << "free pinned memory" << std::endl;
}


void ConcreteInterface::copyTo(void* Dst, const void* Src, size_t Count) {
  std::cout << "copy data from host to device" << std::endl;
}


void ConcreteInterface::copyFrom(void* Dst, const void* Src, size_t Count) {
  std::cout << "copy data from device to host" << std::endl;
}


void ConcreteInterface::copyBetween(void* Dst, const void* Src, size_t Count) {
  std::cout << "copy data within a device" << std::endl;
}


void ConcreteInterface::copy2dArrayTo(void *Dst,
                                      size_t Dpitch,
                                      const void *Src,
                                      size_t Spitch,
                                      size_t Width,
                                      size_t Height) {
  std::cout << "copy data from host to device as a 2D array" << std::endl;
}


void ConcreteInterface::copy2dArrayFrom(void *Dst,
                                        size_t Dpitch,
                                        const void *Src,
                                        size_t Spitch,
                                        size_t Width,
                                        size_t Height) {
  std::cout << "copy data from device to host as a 2D array" << std::endl;
}


void ConcreteInterface::compareDataWithHost(const real *HostPtr,
                                            const real *DevPtr,
                                            const size_t NumElements,
                                            const char *ArrayName) {
  std::cout << "comparing data between host and device" << std::endl;
};


void ConcreteInterface::initialize() {
  std::cout << "init a dummy device" << std::endl;
};


void ConcreteInterface::finalize() {
  m_HasFinalized = true;
  std::cout << "finilize a dummy device" << std::endl;
};