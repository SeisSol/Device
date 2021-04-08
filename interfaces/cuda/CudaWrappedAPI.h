#ifndef DEVICE_CUDAINTERFACE_H
#define DEVICE_CUDAINTERFACE_H

#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "AbstractAPI.h"
#include "Statistics.h"

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

  void * getDefaultStream() override;
  void *getNextCircularStream() override;
  void resetCircularStreamCounter() override;
  size_t getCircularStreamSize() override;
  void syncStreamFromCircularBuffer(void *streamPtr) override;
  void syncCircularBuffer() override;
  void fastStreamsSync() override;

  void initialize() override;
  void finalize() override;
  void putProfilingMark(const std::string &name, ProfilingColors color) override;
  void popLastProfilingMark() override;

private:
  int m_currentDeviceId = 0;

  std::vector<cudaStream_t> m_circularStreamBuffer{};
  size_t m_circularStreamCounter{0};

  char *m_stackMemory = nullptr;
  size_t m_stackMemByteCounter = 0;
  size_t m_maxStackMem = 1024 * 1024 * 1024; //!< 1GB in bytes
  std::stack<size_t> m_stackMemMeter{};

  Statistics m_statistics{};
  std::unordered_map<void *, size_t> m_memToSizeMap{{nullptr, 0}};
};
} // namespace device

#endif // DEVICE_CUDAINTERFACE_H