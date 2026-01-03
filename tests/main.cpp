// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "gtest/gtest.h"
#include <device.h>

using namespace device;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  DeviceInstance& device = DeviceInstance::getInstance();
  device.api->setDevice(0);
  device.api->initialize();

  const auto result = RUN_ALL_TESTS();

  device.api->syncDevice();
  device.api->finalize();

  return result;
}
