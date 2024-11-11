// SPDX-FileCopyrightText: 2021-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DeviceStack.h"

namespace device {
DeviceStack::DeviceStack(cl::sycl::queue &deviceQueue)
    : deviceQueue{deviceQueue}, stackMemoryMeter{std::stack<size_t>{}}, stackMemory{nullptr}, stackMemByteCounter{0},
      maxStackMemory{DEFAULT_STACK_MEMORY} {}

DeviceStack::~DeviceStack() {
  if (stackMemory != nullptr) {
    cl::sycl::free(this->stackMemory, this->deviceQueue);
  }
}

void DeviceStack::initMemory() {
  if (stackMemory != nullptr) {
    auto name = deviceQueue.get_device().get_info<cl::sycl::info::device::name>();
    throw std::invalid_argument(name.append(" - Illegal State: stack memory already initialized"));
  }

  char *valueString = getenv("DEVICE_STACK_MEM_SIZE");
  if (valueString) {
    double requestedStackMem = std::stod(std::string(valueString));
    this->maxStackMemory = DEFAULT_STACK_MEMORY * requestedStackMem;
  }

  this->stackMemory = static_cast<char *>(malloc_device(this->maxStackMemory, this->deviceQueue));
}

char *DeviceStack::getStackMemory(size_t requestedBytes) {
  if ((stackMemByteCounter + requestedBytes) >= maxStackMemory) {
    auto name = deviceQueue.get_device().get_info<cl::sycl::info::device::name>();
    throw std::invalid_argument(name.append(": Stack out of memory"));
  }

  char *mem = &stackMemory[stackMemByteCounter];
  stackMemByteCounter += requestedBytes;
  stackMemoryMeter.push(requestedBytes);
  return mem;
}

void DeviceStack::popStackMemory() {
  stackMemByteCounter -= stackMemoryMeter.top();
  stackMemoryMeter.pop();
}

size_t DeviceStack::getStackMemByteCounter() { return this->stackMemByteCounter; }

} // namespace device

