#ifndef DEVICE_HIPINTERFACE_H
#define DEVICE_HIPINTERFACE_H

#include "AbstractAPI.h"
#include "Statistics.h"
#include "hip/hip_runtime.h"
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>


namespace device {
class ConcreteAPI : public AbstractAPI {
public:
  void setDevice(int deviceId) override;
  int getNumDevices() override;
  unsigned getMaxThreadBlockSize() override;
  unsigned getMaxSharedMemSize() override;
  unsigned getGlobMemAlignment() override;
  std::string getDeviceInfoAsText(int deviceId) override;
  void synchDevice() override;
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
  void copy2dArrayTo(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                     size_t height) override;
  void copy2dArrayFrom(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                       size_t height) override;
  void prefetchUnifiedMemTo(Destination type, const void *devPtr, size_t count,
                            void *streamPtr) override;

  size_t getMaxAvailableMem() override;
  size_t getCurrentlyOccupiedMem() override;
  size_t getCurrentlyOccupiedUnifiedMem() override;

  void * getDefaultStream() override;
  void syncDefaultStreamWithHost() override;

  void *getNextCircularStream() override;
  void resetCircularStreamCounter() override;
  size_t getCircularStreamSize() override;
  void syncStreamFromCircularBufferWithHost(void *userStream) override;
  void syncCircularBuffer() override;
  void syncCircularBuffersWithHost() override;

  void forkCircularStreamsFromDefault() override;
  void joinCircularStreamsToDefault() override;
  bool isCircularStreamsJoinedWithDefault() override;

  bool isCapableOfGraphCapturing() { return false; }
  void streamBeginCapture() override {}
  void streamEndCapture() override {}
  deviceGraphHandle getGraphInstance() { return deviceGraphHandle{}; }
  void launchGraph(deviceGraphHandle graphHandle) {}
  void syncGraph(deviceGraphHandle graphHandle) {}

  void initialize() override;
  void finalize() override;
  void putProfilingMark(const std::string &name, ProfilingColors color) override;
  void popLastProfilingMark() override;

  void createAlStreamsAndEvents();

private:
  int m_currentDeviceId = 0;

  hipStream_t m_defaultStream{};
  hipEvent_t m_defaultStreamEvent{};

  std::vector<hipStream_t> m_circularStreamBuffer{};
  std::vector<hipEvent_t> m_circularStreamEvents{};
  bool m_isCircularStreamsForked{false};
  size_t m_circularStreamCounter{0};

  char *m_stackMemory = nullptr;
  size_t m_stackMemByteCounter = 0;
  size_t m_maxStackMem = 1024 * 1024 * 1024; //!< 1GB in bytes
  std::stack<size_t> m_stackMemMeter{};

  Statistics m_statistics{};
  std::unordered_map<void *, size_t> m_memToSizeMap{{nullptr, 0}};
};
} // namespace device

#endif // DEVICE_HIPINTERFACE_H
