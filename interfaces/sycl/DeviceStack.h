#ifndef DEVICE_STACK_H
#define DEVICE_STACK_H

#include <CL/sycl.hpp>
#include <stack>
#include <string.h>

#define DEFAULT_STACK_MEMORY 1024 * 1024 * 1024


namespace device {
class DeviceStack {
public:
  DeviceStack(cl::sycl::queue &deviceQueue);
  ~DeviceStack();

  void initMemory();
  char *getStackMemory(size_t requestedBytes);
  void popStackMemory();

  size_t getStackMemByteCounter();

private:
  cl::sycl::queue &deviceQueue;
  std::stack<size_t> stackMemoryMeter;
  char *stackMemory;
  size_t stackMemByteCounter;
  size_t maxStackMemory;
};

} // namespace device

#endif
