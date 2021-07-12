#ifndef DEVICE_ABSTRACT_API_H
#define DEVICE_ABSTRACT_API_H

#include "DataTypes.h"
#include <cstdlib>
#include <string>

namespace device {

enum class Destination { Host, CurrentDevice };
enum class StreamType { Blocking, NonBlocking };
enum class ProfilingColors : uint32_t {
  Black = 0x000000,
  White = 0xFFFFFF,
  Red = 0xFF0000,
  Lime = 0x00FF00,
  Blue = 0x0000FF,
  Yellow = 0xFFFF00,
  Cyan = 0x00FFFF,
  Magenta = 0xFF00FF,
  Count = 8
};
struct AbstractAPI {
  virtual ~AbstractAPI() = default;

  virtual void setDevice(int deviceId) = 0;
  virtual int getNumDevices() = 0;
  virtual unsigned getMaxThreadBlockSize() = 0;
  virtual unsigned getMaxSharedMemSize() = 0;
  virtual unsigned getGlobMemAlignment() = 0;
  virtual std::string getDeviceInfoAsText(int deviceId) = 0;
  virtual void synchDevice() = 0;
  virtual void checkOffloading() = 0;

  virtual void allocateStackMem() = 0;
  virtual void *allocGlobMem(size_t size) = 0;
  virtual void *allocUnifiedMem(size_t size) = 0;
  virtual void *allocPinnedMem(size_t size) = 0;
  virtual char *getStackMemory(size_t requestedBytes) = 0;
  virtual void freeMem(void *devPtr) = 0;
  virtual void freePinnedMem(void *devPtr) = 0;
  virtual void popStackMemory() = 0;
  virtual std::string getMemLeaksReport() = 0;

  virtual void copyTo(void *dst, const void *src, size_t count) = 0;
  virtual void copyFrom(void *dst, const void *src, size_t count) = 0;
  virtual void copyBetween(void *dst, const void *src, size_t count) = 0;
  virtual void copyToAsync(void *dst, const void *src, size_t count, void* streamPtr) = 0;
  virtual void copyFromAsync(void *dst, const void *src, size_t count, void* streamPtr) = 0;
  virtual void copyBetweenAsync(void *dst, const void *src, size_t count, void* streamPtr) = 0;

  virtual void copy2dArrayTo(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                             size_t height) = 0;
  virtual void copy2dArrayFrom(void *dst, size_t dpitch, const void *src, size_t spitch,
                               size_t width, size_t height) = 0;
  virtual void prefetchUnifiedMemTo(Destination type, const void *devPtr, size_t count,
                                    void *streamPtr) = 0;

  virtual size_t getMaxAvailableMem() = 0;
  virtual size_t getCurrentlyOccupiedMem() = 0;
  virtual size_t getCurrentlyOccupiedUnifiedMem() = 0;


  virtual void *getDefaultStream() = 0;
  virtual void syncDefaultStreamWithHost() = 0;

  virtual void *getNextCircularStream() = 0;
  virtual void resetCircularStreamCounter() = 0;
  virtual size_t getCircularStreamSize() = 0;
  virtual void syncStreamFromCircularBufferWithHost(void *userStream) = 0;
  virtual void syncCircularBuffersWithHost() = 0;

  virtual void forkCircularStreamsFromDefault() = 0;
  virtual void joinCircularStreamsToDefault() = 0;
  virtual bool isCircularStreamsJoinedWithDefault() = 0;


  virtual bool isCapableOfGraphCapturing() = 0;
  virtual void streamBeginCapture() = 0;
  virtual void streamEndCapture() = 0;
  virtual deviceGraphHandle getLastGraphHandle() = 0;
  virtual void launchGraph(deviceGraphHandle graphHandle) = 0;
  virtual void syncGraph(deviceGraphHandle graphHandle) = 0;

  virtual void initialize() = 0;
  virtual void finalize() = 0;
  virtual void putProfilingMark(const std::string &name, ProfilingColors color) = 0;
  virtual void popLastProfilingMark() = 0;

  bool hasFinalized() { return m_isFinalized; }

protected:
  bool m_isFinalized = false;
};
} // namespace device

#endif // DEVICE_ABSTRACT_API_H
