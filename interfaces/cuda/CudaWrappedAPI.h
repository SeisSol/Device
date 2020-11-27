#ifndef DEVICE_CUDAINTERFACE_H
#define DEVICE_CUDAINTERFACE_H

#include <stack>
#include <unordered_map>
#include <string>
#include <vector>

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
    void streamBatchedData(real **BaseSrcPtr, real **BaseDstPtr, unsigned ElementSize, unsigned NumElements) override;
    void accumulateBatchedData(real **BaseSrcPtr, real **BaseDstPtr, unsigned ElementSize, unsigned NumElements) override;
    void prefetchUnifiedMemTo(Destination Type, const void* DevPtr, size_t Count, void* streamPtr) override;

    size_t getMaxAvailableMem() override;
    size_t getCurrentlyOccupiedMem() override;
    size_t getCurrentlyOccupiedUnifiedMem() override;

    void* getRawCurrentComputeStream() override { return static_cast<void*>(&m_CurrentComputeStream);}
    void* getNextCircularStream() override;
    void resetCircularStreamCounter() override;
    size_t getCircularStreamSize() override;
    void syncStreamFromCircularBuffer(void* streamPtr) override;
    void syncCircularBuffer() override;
    void fastStreamsSync() override;


    void compareDataWithHost(const real *HostPtr,
                             const real *DevPtr,
                             const size_t NumElements,
                             const std::string& DataName) override;
    void scaleArray(real *DevArray, const real Scalar, const size_t NumElements) override;

    void touchMemory(real *Ptr, size_t Size, bool Clean) override;
    void touchBatchedMemory(real **BasePtr, unsigned ElementSize, unsigned NumElements, bool Clean) override;

    void initialize() override;
    void finalize() override;
    void putProfilingMark(const std::string &Name, ProfilingColors Color) override;
    void popLastProfilingMark() override;

  private:
    int m_CurrentDeviceId = 0;

    std::vector<cudaStream_t> m_circularStreamBuffer{};
    size_t m_circularStreamCounter{0};
    cudaStream_t m_CurrentComputeStream = 0;

    char* m_StackMemory = nullptr;
    size_t m_StackMemByteCounter = 0;
    size_t m_MaxStackMem = 1024 * 1024 * 1024;  //!< 1GB in bytes
    std::stack<size_t> m_StackMemMeter{};

    Statistics m_Statistics{};
    std::unordered_map<void*, size_t> m_MemToSizeMap{{nullptr, 0}};
  };
}

#endif //DEVICE_CUDAINTERFACE_H