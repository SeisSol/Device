#include "device.h"
#include "DataTypes.h"
#include <iostream>
#include <stdlib.h>

using namespace device;

int main(int Argc, char *Argv[]) {
  const size_t SIZE = 1024;
  real *InputArray = new real[SIZE];
  real *OutputArray = new real[SIZE];

  DeviceInstance &Device = DeviceInstance::getInstance();

  // set up the first device
  const int NUM_DEVICES = Device.api->getNumDevices();
  std::cout << "Num. devices available: " << NUM_DEVICES << '\n';
  if (NUM_DEVICES > 0) {
    Device.api->setDevice(0);
  }

  // print some device info
  std::string DeviceInfo(Device.api->getDeviceInfoAsText(0));
  std::cout << DeviceInfo << std::endl;

  // allocate mem. on a device
  real *DInputArray = static_cast<real *>(Device.api->allocGlobMem(sizeof(real) * SIZE));
  real *DOutputArray = static_cast<real *>(Device.api->allocGlobMem(sizeof(real) * SIZE));

  // copy data into a device
  Device.api->copyTo(DInputArray, InputArray, sizeof(real) * SIZE);

  // call a kernel
  Device.api->checkOffloading();
  Device.api->synchDevice();

  // copy data from a device
  Device.api->copyFrom(OutputArray, DOutputArray, sizeof(real) * SIZE);

  // deallocate mem. on a device
  Device.api->freeMem(DInputArray);
  Device.api->freeMem(DOutputArray);
  Device.finalize();

  delete[] OutputArray;
  delete[] InputArray;
}