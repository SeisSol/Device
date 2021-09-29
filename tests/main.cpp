#include <device.h>
#include "gtest/gtest.h"

using namespace device;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  DeviceInstance &device = DeviceInstance::getInstance();
  device.api->setDevice(0);
  device.api->initialize();

  return RUN_ALL_TESTS();
}
