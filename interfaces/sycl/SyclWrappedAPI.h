#ifndef DEVICE_SYCLINTERFACE_H
#define DEVICE_SYCLINTERFACE_H

#include "AbstractAPI.h"
#include "Common.h"
#include "DeviceContext.h"
#include "Statistics.h"

#include <CL/sycl.hpp>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>


namespace device {
class ConcreteAPI : public AbstractAPI {

public:
  ConcreteAPI()
      : availableDevices{std::vector<DeviceContext *>{}}, currentDefaultQueue{nullptr}, currentDeviceStack{nullptr},
        currentQueueBuffer{nullptr}, currentStatistics{nullptr}, currentMemoryToSizeMap{nullptr}, deviceInitialized{false},
        currentDeviceId{0}{
    this->initDevices();
  }

  void setDevice(int deviceId) override;

  int getDeviceId() override;
  size_t getLaneSize() override;
  int getNumDevices() override;
  unsigned getMaxThreadBlockSize() override;
  unsigned getMaxSharedMemSize() override;
  unsigned getGlobMemAlignment() override;
  std::string getDeviceInfoAsText(int deviceId) override;
  void syncDevice() override;
  void checkOffloading() override;

  void allocateStackMem() override;
  void *allocGlobMem(size_t size) override;
  void *allocUnifiedMem(size_t size) override;
  void *allocPinnedMem(size_t size) override;
  char *getStackMemory(size_t requestedBytes) override;
  void freeMem(void *devPtr) override;
  void freePinnedMem(void *devPtr) override;
  void popStackMemory() override;
  std::string getMemLeaksReport() override;

  void copyTo(void *dst, const void *src, size_t count) override;
  void copyFrom(void *dst, const void *src, size_t count) override;
  void copyBetween(void *dst, const void *src, size_t count) override;
  void copy2dArrayTo(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) override;
  void copy2dArrayFrom(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) override;
  void prefetchUnifiedMemTo(Destination type, const void *devPtr, size_t count, void *streamPtr) override;

  void copyToAsync(void *dst, const void *src, size_t count, void* streamPtr) override;
  void copyFromAsync(void *dst, const void *src, size_t count, void* streamPtr) override;
  void copyBetweenAsync(void *dst, const void *src, size_t count, void* streamPtr) override;

  size_t getMaxAvailableMem() override;
  size_t getCurrentlyOccupiedMem() override;
  size_t getCurrentlyOccupiedUnifiedMem() override;

  void *getDefaultStream() override;
  void syncDefaultStreamWithHost() override;

  void *getNextCircularStream() override;
  void resetCircularStreamCounter() override;
  size_t getCircularStreamSize() override;
  void syncStreamFromCircularBufferWithHost(void* streamPtr) override;
  void syncCircularBuffersWithHost() override;

  void forkCircularStreamsFromDefault() override;
  void joinCircularStreamsToDefault() override;
  bool isCircularStreamsJoinedWithDefault() override;

  bool isCapableOfGraphCapturing() override { return false; };
  void streamBeginCapture() override {};
  void streamEndCapture() override {};
  DeviceGraphHandle getLastGraphHandle() override {
      return DeviceGraphHandle{};
  };
  void launchGraph(DeviceGraphHandle graphHandle) override {};
  void syncGraph(DeviceGraphHandle graphHandle) override {};

  void initialize() override;
  void finalize() override;
  void putProfilingMark(const std::string &name, ProfilingColors color) override;
  void popLastProfilingMark() override;

private:
  std::vector<DeviceContext *> availableDevices;

  cl::sycl::queue *currentDefaultQueue;
  bool isCircularStreamsForked{false};

  DeviceStack *currentDeviceStack;
  DeviceCircularQueueBuffer *currentQueueBuffer;
  Statistics *currentStatistics;
  std::unordered_map<void *, size_t> *currentMemoryToSizeMap;

  void initDevices();

  std::string getCurrentDeviceInfoAsText();
  std::string getDeviceInfoAsText(cl::sycl::device dev);

  bool deviceInitialized;
  int currentDeviceId;
};
} // namespace device

#endif
