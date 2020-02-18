#include <iostream>
#include <stdlib.h>

#include "device.h"

using namespace device;

int main(int argc, char* argv[]) {
  const size_t Size = 1024;
  real *InputArray = new real[Size];
  real *OutputArray = new real[Size];


  Device& device = Device::getInstance();
  // set up the first device
  const int NumDevices = device.api->getNumDevices();
  std::cout << "Num. devices available: " << NumDevices << '\n';
  if (NumDevices > 0) {
    device.api->setDevice(0);
  }

  // print some device info
  std::string DeviceInfo(device.api->getDeviceInfoAsText(0));
  std::cout << DeviceInfo << std::endl;

  device.api->checkOffloading();

  /*
  real *ScratchMem = reinterpret_cast<real*>(device.api->getTempMemory(1024 * sizeof(real)));
  device.api->freeTempMemory();

  unsigned StreamId = device.api->createStream();
  std::cout << "given stream id: " << StreamId << std::endl;
  */

  // allocate mem. on a device
  real *d_InputArray = static_cast<real*>(device.api->allocGlobMem(sizeof(real) * Size));
  real *d_OutputArray = static_cast<real*>(device.api->allocGlobMem(sizeof(real) * Size));

  // copy data into a device
  device.api->copyTo(d_InputArray, InputArray, sizeof(real) * Size);

  // call a kernel
  device.api->synchDevice();

  // copy data from a device
  device.api->copyFrom(OutputArray, d_OutputArray, sizeof(real) * Size);

  // deallocate mem. on a device
  device.api->freeMem(d_InputArray);
  device.api->freeMem(d_OutputArray);
  device.finalize();

  delete [] OutputArray;
  delete [] InputArray;
}