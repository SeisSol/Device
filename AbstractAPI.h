// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_ABSTRACTAPI_H_
#define SEISSOLDEVICE_ABSTRACTAPI_H_

#include "DataTypes.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace device {

enum class Destination { Host, CurrentDevice };
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
  virtual int getDeviceId() = 0;

  virtual int getNumDevices() = 0;
  virtual unsigned getGlobMemAlignment() = 0;
  virtual std::string getDeviceInfoAsText(int deviceId) = 0;
  virtual void syncDevice() = 0;

  virtual std::string getApiName() = 0;
  virtual std::string getDeviceName(int deviceId) = 0;
  virtual std::string getPciAddress(int deviceId) = 0;

  virtual void *allocGlobMem(size_t size, bool compress = false) = 0;
  virtual void *allocUnifiedMem(size_t size, bool compress = false, Destination hint = Destination::CurrentDevice) = 0;
  virtual void *allocPinnedMem(size_t size, bool compress = false, Destination hint = Destination::Host) = 0;
  virtual void freeGlobMem(void *devPtr) = 0;
  virtual void freeUnifiedMem(void *devPtr) = 0;
  virtual void freePinnedMem(void *devPtr) = 0;
  virtual std::string getMemLeaksReport() = 0;

  virtual void *allocMemAsync(size_t size, void* streamPtr) = 0;
  virtual void freeMemAsync(void *devPtr, void* streamPtr) = 0;

  virtual void pinMemory(void* ptr, size_t size) = 0;
  virtual void unpinMemory(void* ptr) = 0;
  virtual void* devicePointer(void* ptr) = 0;

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

  virtual bool isCapableOfGraphCapturing() = 0;
  virtual DeviceGraphHandle streamBeginCapture(std::vector<void*>& streamPtrs) = 0;
  virtual void streamEndCapture(DeviceGraphHandle handle) = 0;
  virtual void launchGraph(DeviceGraphHandle graphHandle, void* streamPtr) = 0;

  virtual void* createStream(double priority = NAN) = 0;
  virtual void destroyGenericStream(void* streamPtr) = 0;
  virtual void syncStreamWithHost(void* streamPtr) = 0;
  virtual bool isStreamWorkDone(void* streamPtr) = 0;
  virtual void syncStreamWithEvent(void* streamPtr, void* eventPtr) = 0;
  virtual void streamHostFunction(void* streamPtr, const std::function<void()>& function) = 0;

  virtual void streamWaitMemory(void* streamPtr, uint32_t* location, uint32_t value) = 0;

  virtual void* createEvent() = 0;
  virtual void destroyEvent(void* eventPtr) = 0;
  virtual void syncEventWithHost(void* eventPtr) = 0;
  virtual bool isEventCompleted(void* eventPtr) = 0;
  virtual void recordEventOnHost(void* eventPtr) = 0;
  virtual void recordEventOnStream(void* eventPtr, void* streamPtr) = 0;

  virtual bool isUnifiedMemoryDefault() = 0;

  virtual void initialize() = 0;
  virtual void finalize() = 0;
  virtual void profilingMessage(const std::string& message) = 0;
  virtual void putProfilingMark(const std::string &name, ProfilingColors color) = 0;
  virtual void popLastProfilingMark() = 0;

  virtual void setupPrinting(int rank) = 0;

  bool hasFinalized() { return m_isFinalized; }

protected:
  bool m_isFinalized = false;
  std::mutex apiMutex;
};
} // namespace device


#endif // SEISSOLDEVICE_ABSTRACTAPI_H_

