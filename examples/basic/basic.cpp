#include "DataTypes.h"
#include "device.h"
#include <iostream>
#include <stdlib.h>

using namespace device;

int main(int argc, char *argv[]) {
  const size_t size = 1024;
  real *inputArray = new real[size];
  real *outputArray = new real[size];

  DeviceInstance &device = DeviceInstance::getInstance();

  // set up the first device
  const int numDevices = device.api->getNumDevices();
  std::cout << "Num. devices available: " << numDevices << '\n';
  if (numDevices > 0) {
    device.api->setDevice(0);
  }

  // print some device info
  std::string deviceInfo(device.api->getDeviceInfoAsText(0));
  std::cout << deviceInfo << std::endl;

  // allocate mem. on a device
  real *dInputArray = static_cast<real *>(device.api->allocGlobMem(sizeof(real) * size));
  real *dOutputArray = static_cast<real *>(device.api->allocGlobMem(sizeof(real) * size));

  // copy data into a device
  device.api->copyTo(dInputArray, inputArray, sizeof(real) * size);

  // call a kernel
  device.api->checkOffloading();
  device.api->synchDevice();

  // copy data from a device
  device.api->copyFrom(outputArray, dOutputArray, sizeof(real) * size);

  // deallocate mem. on a device
  device.api->freeMem(dInputArray);
  device.api->freeMem(dOutputArray);

  std::cout << device.api->getMemLeaksReport();

  device.finalize();

  delete[] outputArray;
  delete[] inputArray;
}