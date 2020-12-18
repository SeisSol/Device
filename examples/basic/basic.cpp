#include "DataTypes.h"
#include "device.h"
#include <iostream>
#include <stdlib.h>

using namespace device;

int main(int argc, char *argv[]) {
  const size_t SIZE = 1024;
  real *inputArray = new real[SIZE];
  real *outputArray = new real[SIZE];

  DeviceInstance &device = DeviceInstance::getInstance();

  // set up the first device
  const int NUM_DEVICES = device.api->getNumDevices();
  std::cout << "Num. devices available: " << NUM_DEVICES << '\n';
  if (NUM_DEVICES > 0) {
    device.api->setDevice(0);
  }

  // print some device info
  std::string deviceInfo(device.api->getDeviceInfoAsText(0));
  std::cout << deviceInfo << std::endl;

  // allocate mem. on a device
  real *dInputArray = static_cast<real *>(device.api->allocGlobMem(sizeof(real) * SIZE));
  real *dOutputArray = static_cast<real *>(device.api->allocGlobMem(sizeof(real) * SIZE));

  // copy data into a device
  device.api->copyTo(dInputArray, inputArray, sizeof(real) * SIZE);

  // call a kernel
  device.api->checkOffloading();
  device.api->synchDevice();

  // copy data from a device
  device.api->copyFrom(outputArray, dOutputArray, sizeof(real) * SIZE);

  // deallocate mem. on a device
  device.api->freeMem(dInputArray);
  device.api->freeMem(dOutputArray);
  device.finalize();

  delete[] outputArray;
  delete[] inputArray;
}