// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common.h"
#include "device.h"

#include <cstdlib>
#include <iostream>

using namespace device;

int main(int argc, char* argv[]) {
  const size_t size = 1024;
  real* inputArray = new real[size];
  real* outputArray = new real[size];
  for (size_t i = 0; i < size; ++i) {
    inputArray[i] = static_cast<real>(i);
    outputArray[i] = 0;
  }

  DeviceInstance& device = DeviceInstance::instance();

  // set up the first device
  const int numDevices = device.api().getNumDevices();
  std::cout << "Num. devices available: " << numDevices << '\n';
  if (numDevices > 0) {
    device.api().setDevice(0);
  }
  device.api().initialize();

  // print some device info
  std::string deviceInfo(device.api().getDeviceInfoAsText(0));
  std::cout << deviceInfo << std::endl;

  std::cout << "alignment: " << device.api().getGlobMemAlignment() << std::endl;
  std::cout << "max available mem: " << device.api().getMaxAvailableMem() << std::endl;

  // allocate mem. on a device
  real* dInputArray = static_cast<real*>(device.api().allocGlobMem(sizeof(real) * size));
  real* dOutputArray = static_cast<real*>(device.api().allocGlobMem(sizeof(real) * size));

  // copy data into a device
  device.api().copyTo(dInputArray, inputArray, sizeof(real) * size);

  // copy data on the device
  device.api().copyBetween(dOutputArray, dInputArray, sizeof(real) * size);
  device.api().syncDevice();

  // copy data from a device
  device.api().copyFrom(outputArray, dOutputArray, sizeof(real) * size);

  size_t mismatches = 0;
  for (size_t i = 0; i < size; ++i) {
    if (outputArray[i] != inputArray[i]) {
      ++mismatches;
    }
  }
  std::cout << "mismatches after the round trip: " << mismatches << std::endl;

  // deallocate mem. on a device
  device.api().freeGlobMem(dInputArray);
  device.api().freeGlobMem(dOutputArray);

  std::cout << device.api().getMemLeaksReport();

  device.finalize();

  delete[] outputArray;
  delete[] inputArray;

  return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
