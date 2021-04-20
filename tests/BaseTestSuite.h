#ifndef DEVICE_SUBMODULE_BASETESTSUITE_H
#define DEVICE_SUBMODULE_BASETESTSUITE_H

#include "device.h"
#include "gtest/gtest.h"
#include <iostream>
#include <random>

using namespace device;
using namespace ::testing;

static bool setUp = false;


class BaseTestSuite : public ::testing::Test
{
public:
    DeviceInstance *device;

  BaseTestSuite() { randomEngine.seed(randomDevice()); }

void SetUp() {
    device = &DeviceInstance::getInstance();
  if(!setUp) {
    device->api->allocateStackMem();
  }
  setUp = true;
}

protected:
std::random_device randomDevice;
std::mt19937 randomEngine;
};


#endif // DEVICE_SUBMODULE_BASETESTSUITE_H
