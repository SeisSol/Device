#ifndef DEVICE_CUDAINTERFACE_H
#define DEVICE_CUDAINTERFACE_H

#include <stack>
#include <unordered_map>

#include "AbstractInterface.h"
#include "Statistics.h"

namespace device {
  class ConcreteInterface : public AbstractInterface {
  public:
    void setDevice(int DeviceId);
    int getNumDevices();
    std::string getDeviceInfoAsText(int DeviceId);
    void synchDevice();
    void checkOffloading();

    void* allocGlobMem(size_t Size);
    void* allocUnifiedMem(size_t Size);
    void* allocPinnedMem(size_t Size);
    char* getStackMemory(unsigned RequestedBytes);
    void freeMem(void *DevPtr);
    void freePinnedMem(void *DevPtr);
    void popStackMemory();

    void copyTo(void* Dst, const void* Src, size_t Count);
    void copyFrom(void* Dst, const void* Src, size_t Count);
    void copyBetween(void* Dst, const void* Src, size_t Count);
    void copy2dArrayTo(void *Dst, size_t Dpitch, const void *Src, size_t Spitch, size_t Width, size_t Height);
    void copy2dArrayFrom(void *Dst, size_t Dpitch, const void *Src, size_t Spitch, size_t Width, size_t Height);
    void streamBatchedData(real **BaseSrcPtr, real **BaseDstPtr, unsigned ElementSize, unsigned NumElements);
    void accumulateBatchedData(real **BaseSrcPtr, real **BaseDstPtr, unsigned ElementSize, unsigned NumElements);


    unsigned createStream();
    void deleteStream(unsigned StreamId);
    void deleteAllCreatedStreams();
    void setComputeStream(unsigned StreamId);
    void setDefaultComputeStream();
    void synchAllStreams();


    void compareDataWithHost(const real *HostPtr,
                             const real *DevPtr,
                             const size_t NumElements,
                             const char *ArrayName = nullptr);

    void touchMemory(real *Ptr, size_t Size, bool Clean);
    void touchBatchedMemory(real **BasePtr, unsigned ElementSize, unsigned NumElements, bool Clean);


    void initialize();
    void finalize();
  private:
    cudaStream_t m_CurrentComputeStream;
    cudaStream_t m_DefaultStream = 0;
    std::unordered_map<int, cudaStream_t*> m_IdToStreamMap{};

    char* m_StackMemory = nullptr;
    size_t m_StackMemByteCounter = 0;
    size_t m_MaxStackMem = 1024 * 1024 * 1024;  //!< 1GB in bytes
    std::stack<size_t> m_StackMemMeter{};

    Statistics m_Statistics{};
    std::unordered_map<void*, size_t> m_MemToSizeMap{{nullptr, 0}};
  };
}


#endif //DEVICE_CUDAINTERFACE_H
