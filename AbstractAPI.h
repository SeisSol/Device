#ifndef DEVICE_ABSTRACT_API_H
#define DEVICE_ABSTRACT_API_H

#include <stdlib.h>
#include <string>

#include "DataTypes.h"

namespace device {

  enum class Destination {Host, CurrentDevice};
  enum class StreamType {Blocking, NonBlocking};
  enum class ProfilingColors : uint32_t {Black = 0x000000,
                                         White = 0xFFFFFF,
                                         Red = 0xFF0000,
                                         Lime = 0x00FF00,
                                         Blue = 0x0000FF,
                                         Yellow = 0xFFFF00,
                                         Cyan = 0x00FFFF,
                                         Magenta = 0xFF00FF,
                                         Count = 8};

  struct AbstractAPI {

    virtual void setDevice(int DeviceId) = 0;
    virtual int getNumDevices() = 0;
    virtual unsigned getMaxThreadBlockSize() = 0;
    virtual unsigned getMaxSharedMemSize() = 0;
    virtual unsigned getGlobMemAlignment() = 0;
    virtual std::string getDeviceInfoAsText(int DeviceId) = 0;
    virtual void synchDevice() = 0;
    virtual void checkOffloading() = 0;

    virtual void allocateStackMem() = 0;
    virtual void* allocGlobMem(size_t Size) = 0;
    virtual void* allocUnifiedMem(size_t Size) = 0;
    virtual void* allocPinnedMem(size_t Size) = 0;
    virtual char* getStackMemory(size_t RequestedBytes) = 0;
    virtual void freeMem(void *DevPtr) = 0;
    virtual void freePinnedMem(void *DevPtr) = 0;
    virtual void popStackMemory() = 0;
    virtual std::string getMemLeaksReport() = 0;

    virtual void copyTo(void* Dst, const void* Src, size_t Count) = 0;
    virtual void copyFrom(void* Dst, const void* Src, size_t Count) = 0;
    virtual void copyBetween(void* Dst, const void* Src, size_t Count) = 0;
    virtual void copy2dArrayTo(void *Dst, size_t Dpitch, const void *Src, size_t Spitch, size_t Width, size_t Height) = 0;
    virtual void copy2dArrayFrom(void *Dst, size_t Dpitch, const void *Src, size_t Spitch, size_t Width, size_t Height) = 0;
    virtual void streamBatchedData(real **BaseSrcPtr, real **BaseDstPtr, unsigned ElementSize, unsigned NumElements) = 0;
    virtual void accumulateBatchedData(real **BaseSrcPtr, real **BaseDstPtr, unsigned ElementSize, unsigned NumElements) = 0;
    virtual void prefetchUnifiedMemTo(Destination Type, const void* DevPtr, size_t Count, int StreamId = 0) = 0;

    virtual unsigned createStream(StreamType Type = StreamType::Blocking) = 0;
    virtual void deleteStream(unsigned StreamId) = 0;
    virtual void deleteAllCreatedStreams() = 0;
    virtual void setComputeStream(unsigned StreamId) = 0;
    virtual void* getRawCurrentComputeStream() = 0;  // TODO: must be protected
    virtual void setDefaultComputeStream() = 0;
    virtual void synchAllStreams() = 0;

    virtual void compareDataWithHost(const real *HostPtr,
                                     const real *DevPtr,
                                     const size_t NumElements,
                                     const std::string& DataName) = 0;
    virtual void scaleArray(real *DevArray, const real Scalar, const size_t NumElements) = 0;

    virtual void touchMemory(real *Ptr, size_t Size, bool Clean) = 0;
    virtual void touchBatchedMemory(real **BasePtr, unsigned ElementSize, unsigned NumElements, bool Clean) = 0;

    virtual void initialize() = 0;
    virtual void finalize() = 0;
    virtual void putProfilingMark(const std::string &Name, ProfilingColors Color) = 0;
    virtual void popLastProfilingMark() = 0;

    bool hasFinalized() { return m_HasFinalized; }
  protected:
    bool m_HasFinalized = false;
  };
}

#endif //DEVICE_ABSTRACT_API_H
