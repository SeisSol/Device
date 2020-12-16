#ifndef DEVICE_HIPINTERFACE_H
#define DEVICE_HIPINTERFACE_H

#include "hip/hip_runtime.h"
#include <stack>
#include <unordered_map>
#include <string>

#include "AbstractAPI.h"
#include "Statistics.h"

namespace device {
class ConcreteAPI : public AbstractAPI {
public:
  void setDevice(int DeviceId) override;
  int getNumDevices() override;
  unsigned getMaxThreadBlockSize() override;
  unsigned getMaxSharedMemSize() override;
  unsigned getGlobMemAlignment() override;
  std::string getDeviceInfoAsText(int DeviceId) override;
  void synchDevice() override;
  void checkOffloading() override;

  void allocateStackMem() override;
  void* allocGlobMem(size_t Size) override;
  void* allocUnifiedMem(size_t Size) override;
  void* allocPinnedMem(size_t Size) override;
  char* getStackMemory(size_t RequestedBytes) override;
  void freeMem(void *DevPtr) override;
  void freePinnedMem(void *DevPtr) override;
  void popStackMemory() override;
  std::string getMemLeaksReport() override;

  void copyTo(void* Dst, const void* Src, size_t Count) override;
  void copyFrom(void* Dst, const void* Src, size_t Count) override;
  void copyBetween(void* Dst, const void* Src, size_t Count) override;
  void copy2dArrayTo(void *Dst, size_t Dpitch, const void *Src, size_t Spitch, size_t Width, size_t Height) override;
  void copy2dArrayFrom(void *Dst, size_t Dpitch, const void *Src, size_t Spitch, size_t Width, size_t Height) override;
  void prefetchUnifiedMemTo(Destination Type, const void* DevPtr, size_t Count, void* streamPtr) override;

  size_t getMaxAvailableMem() override;
  size_t getCurrentlyOccupiedMem() override;
  size_t getCurrentlyOccupiedUnifiedMem() override;

  void* getNextCircularStream() override;
  void resetCircularStreamCounter() override;
  size_t getCircularStreamSize() override;
  void syncStreamFromCircularBuffer(void* streamPtr) override;
  void syncCircularBuffer() override;
  void fastStreamsSync() override;

  void initialize() override;
  void finalize() override;
  void putProfilingMark(const std::string &Name, ProfilingColors Color) override;
  void popLastProfilingMark() override;

private:
  int m_CurrentDeviceId = 0;

  std::vector<hipStream_t> m_circularStreamBuffer{};
  size_t m_circularStreamCounter{0};

  char* m_StackMemory = nullptr;
  size_t m_StackMemByteCounter = 0;
  size_t m_MaxStackMem = 1024 * 1024 * 1024;  //!< 1GB in bytes
  std::stack<size_t> m_StackMemMeter{};

  Statistics m_Statistics{};
  std::unordered_map<void*, size_t> m_MemToSizeMap{{nullptr, 0}};
};
} // namespace device

#endif //DEVICE_HIPINTERFACE_H
