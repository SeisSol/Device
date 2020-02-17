#ifndef DEVICE_H
#define DEVICE_H

#include <iostream>

#include "AbstractInterface.h"

namespace device {
  // Device -> Singleton
  class Device {
  public:
    Device(const Device&) = delete;
    Device &operator=(const Device&) = delete;

    static Device& getInstance() {
      static Device Instance;
      return Instance;
    }

    ~Device() {
      this->finalize();
    }

    void finalize();

    AbstractInterface *api = nullptr;

  private:
    Device();
  };
}
#endif  //SEISSOL_DEVICE_H