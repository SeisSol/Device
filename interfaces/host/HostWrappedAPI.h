// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HOST_HOSTWRAPPEDAPI_H_
#define SEISSOLDEVICE_INTERFACES_HOST_HOSTWRAPPEDAPI_H_

#include "AbstractAPI.h"
#include "Common.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace device {

/**
 * The device API, implemented on the host.
 *
 * There is exactly one device: the host itself. Device memory is host memory, and all work runs
 * synchronously on the calling thread; that is, every operation has completed once its call
 * returns. Streams and events only serve as handles; waiting on them returns immediately.
 */
class ConcreteAPI : public AbstractAPI {
  public:
  void setDevice(int deviceId) override;
  int getDeviceId() override;

  int getNumDevices() override;
  unsigned getGlobMemAlignment() override;
  std::string getDeviceInfoAsText(int deviceId) override;
  void syncDevice() override;

  std::string getApiName() override;
  std::string getDeviceName(int deviceId) override;
  std::string getPciAddress(int deviceId) override;

  void* allocGlobMem(size_t size, bool compress) override;
  void* allocUnifiedMem(size_t size, bool compress, Destination hint) override;
  void* allocPinnedMem(size_t size, bool compress, Destination hint) override;
  void freeGlobMem(void* devPtr) override;
  void freeUnifiedMem(void* devPtr) override;
  void freePinnedMem(void* devPtr) override;
  std::string getMemLeaksReport() override;

  void* allocMemAsync(size_t size, void* streamPtr) override;
  void freeMemAsync(void* devPtr, void* streamPtr) override;

  void pinMemory(void* ptr, size_t size) override;
  void unpinMemory(void* ptr) override;
  void* devicePointer(void* ptr) override;

  void copyTo(void* dst, const void* src, size_t count) override;
  void copyFrom(void* dst, const void* src, size_t count) override;
  void copyBetween(void* dst, const void* src, size_t count) override;
  void copyToAsync(void* dst, const void* src, size_t count, void* streamPtr) override;
  void copyFromAsync(void* dst, const void* src, size_t count, void* streamPtr) override;
  void copyBetweenAsync(void* dst, const void* src, size_t count, void* streamPtr) override;

  void copy2dArrayTo(void* dst,
                     size_t dpitch,
                     const void* src,
                     size_t spitch,
                     size_t width,
                     size_t height) override;
  void copy2dArrayFrom(void* dst,
                       size_t dpitch,
                       const void* src,
                       size_t spitch,
                       size_t width,
                       size_t height) override;
  void prefetchUnifiedMemTo(Destination type,
                            const void* devPtr,
                            size_t count,
                            void* streamPtr) override;

  size_t getMaxAvailableMem() override;
  size_t getCurrentlyOccupiedMem() override;
  size_t getCurrentlyOccupiedUnifiedMem() override;

  void* getDefaultStream() override;
  void syncDefaultStreamWithHost() override;

  bool isCapableOfGraphCapturing() override;
  DeviceGraphHandle streamBeginCapture(const std::vector<void*>& streamPtrs) override;
  void streamEndCapture(const DeviceGraphHandle& handle) override;
  void launchGraph(const DeviceGraphHandle& graphHandle, void* streamPtr) override;

  bool isCapableOfGraphNodes() override;
  DeviceGraphHandle graphCreate() override;
  void graphBeginNode(const DeviceGraphHandle& graphHandle,
                      const std::vector<DeviceGraphNodeHandle>& dependencies,
                      void* streamPtr) override;
  DeviceGraphNodeHandle graphEndNode(const DeviceGraphHandle& graphHandle,
                                     void* streamPtr) override;
  void graphInstantiate(const DeviceGraphHandle& graphHandle) override;

  void* createStream(double priority) override;
  void destroyGenericStream(void* streamPtr) override;
  void syncStreamWithHost(void* streamPtr) override;
  bool isStreamWorkDone(void* streamPtr) override;
  void syncStreamWithEvent(void* streamPtr, void* eventPtr) override;
  void streamHostFunction(void* streamPtr, const std::function<void()>& function) override;

  void streamWaitMemory(void* streamPtr, uint32_t* location, uint32_t value) override;

  void* createEvent(bool withTiming) override;
  void destroyEvent(void* eventPtr) override;
  void syncEventWithHost(void* eventPtr) override;
  bool isEventCompleted(void* eventPtr) override;
  void recordEventOnHost(void* eventPtr) override;
  void recordEventOnStream(void* eventPtr, void* streamPtr) override;
  double timespanEvents(void* eventPtrStart, void* eventPtrEnd) override;

  bool isUnifiedMemoryDefault() override;

  void initialize() override;
  void finalize() override;
  void profilingMessage(const std::string& message) override;
  void putProfilingMark(const std::string& name, ProfilingColors color) override;
  void popLastProfilingMark() override;

  void setupPrinting(int rank) override;

  private:
  // a stream only needs an identity
  struct Stream {};

  struct Event {
    std::chrono::steady_clock::time_point recorded{};
  };

  void* allocate(size_t size, bool unified);
  void deallocate(void* ptr);

  static Event* toEvent(void* eventPtr);

  device::StatusT status{false};

  Stream defaultStream{};
  std::unordered_map<void*, std::unique_ptr<Stream>> genericStreams{};

  // the same bookkeeping as in the other backends
  size_t allocatedMemBytes{0};
  size_t allocatedUnifiedMemBytes{0};
  size_t deallocatedMemBytes{0};
  std::unordered_map<void*, size_t> memToSizeMap{};
};
} // namespace device

#endif // SEISSOLDEVICE_INTERFACES_HOST_HOSTWRAPPEDAPI_H_
