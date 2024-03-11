#ifndef DEVICE_HIPINTERFACE_H
#define DEVICE_HIPINTERFACE_H

#include "AbstractAPI.h"
#include "Statistics.h"
#include "Common.h"

#include "hip/hip_runtime.h"
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <cassert>


namespace device {
class ConcreteAPI : public AbstractAPI {
public:
  ConcreteAPI();
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

  std::string getApiName() override;
  std::string getDeviceName(int deviceId) override;
  std::string getPciAddress(int deviceId) override;

  void copyTo(void *dst, const void *src, size_t count) override;
  void copyFrom(void *dst, const void *src, size_t count) override;
  void copyBetween(void *dst, const void *src, size_t count) override;
  void copyToAsync(void *dst, const void *src, size_t count, void* streamPtr) override;
  void copyFromAsync(void *dst, const void *src, size_t count, void* streamPtr) override;
  void copyBetweenAsync(void *dst, const void *src, size_t count, void* streamPtr) override;



  void copy2dArrayTo(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                     size_t height) override;
  void copy2dArrayFrom(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                       size_t height) override;
  void prefetchUnifiedMemTo(Destination type, const void *devPtr, size_t count,
                            void *streamPtr) override;

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

  bool isCapableOfGraphCapturing() override;
  void streamBeginCapture(std::vector<void*>& streamPtrs) override;
  void streamEndCapture() override;
  DeviceGraphHandle getLastGraphHandle() override;
  void launchGraph(DeviceGraphHandle graphHandle, void* streamPtr) override;

  void* createGenericStream() override;
  void destroyGenericStream(void* streamPtr) override;
  void syncStreamWithHost(void* streamPtr) override;
  bool isStreamWorkDone(void* streamPtr) override;
  void syncStreamWithEvent(void* streamPtr, void* eventPtr) override;
  void streamHostFunction(void* streamPtr, const std::function<void()>& function) override;

  void* createEvent() override;
  void destroyEvent(void* eventPtr) override;
  void syncEventWithHost(void* eventPtr) override;
  bool isEventCompleted(void* eventPtr) override;
  void recordEventOnHost(void* eventPtr) override;
  void recordEventOnStream(void* eventPtr, void* streamPtr) override;

  void initialize() override;
  void finalize() override;
  void putProfilingMark(const std::string &name, ProfilingColors color) override;
  void popLastProfilingMark() override;

  bool isUnifiedMemoryDefault() override;

private:
  void createCircularStreamAndEvents();

  device::StatusT status{false};
  int currentDeviceId{-1};

  bool usmDefault{false};

  hipStream_t defaultStream{nullptr};
  hipEvent_t defaultStreamEvent{};

  std::vector<hipStream_t> circularStreamBuffer{};
  std::vector<hipEvent_t> circularStreamEvents{};
  std::unordered_set<hipStream_t> genericStreams{};


  bool isCircularStreamsForked{false};
  size_t circularStreamCounter{0};

  struct GraphDetails {
    hipGraph_t graph;
    hipGraphExec_t instance;
    void* streamPtr;
    bool ready{false};
  };
  std::vector<GraphDetails> graphs;

  char *stackMemory = nullptr;
  size_t stackMemByteCounter = 0;
  size_t maxStackMem = 1024 * 1024 * 1024; //!< 1GB in bytes
  std::stack<size_t> stackMemMeter{};

  Statistics statistics{};
  std::unordered_map<void *, size_t> memToSizeMap{{nullptr, 0}};
};
} // namespace device

#endif // DEVICE_HIPINTERFACE_H
