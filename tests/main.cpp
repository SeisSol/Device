// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include <device.h>
#include "gtest/gtest.h"

using namespace device;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  DeviceInstance &device = DeviceInstance::instance();
  device.api().setDevice(0);
  device.api().initialize();

  return RUN_ALL_TESTS();
}

