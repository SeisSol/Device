// SPDX-FileCopyrightText: 2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_TESTS_BASETESTSUITE_H_
#define SEISSOLDEVICE_TESTS_BASETESTSUITE_H_

#include "device.h"

#include "gtest/gtest.h"
#include <iostream>
#include <random>

using namespace device;
using namespace ::testing;

class BaseTestSuite : public ::testing::Test {
  public:
  DeviceInstance* device{nullptr};

  BaseTestSuite() { randomEngine.seed(randomDevice()); }

  void SetUp() override { device = &DeviceInstance::getInstance(); }

  protected:
  std::random_device randomDevice;
  std::mt19937 randomEngine;
};

#endif // SEISSOLDEVICE_TESTS_BASETESTSUITE_H_
