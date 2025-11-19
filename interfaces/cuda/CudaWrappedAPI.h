// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_CUDA_CUDAWRAPPEDAPI_H_
#define SEISSOLDEVICE_INTERFACES_CUDA_CUDAWRAPPEDAPI_H_

#include <stack>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#include <unordered_set>
#include <cassert>
#include <cstdint>

#include "AbstractAPI.h"
#include "Statistics.h"
#include "Common.h"
#include "cuda.h"

namespace device {
class ConcreteAPI : public AbstractAPI {
public:
  ConcreteAPI();
  void setDevice(int deviceId) override;

  int getNumDevices() override;
  int getDeviceId() override;
  unsigned getGlobMemAlignment() override;
  std::string getDeviceInfoAsText(int deviceId) override;
  void syncDevice() override;

  void *allocGlobMem(size_t size, bool compress) override;
  void *allocUnifiedMem(size_t size, bool compress, Destination hint) override;
  void *allocPinnedMem(size_t size, bool compress, Destination hint) override;
  void freeGlobMem(void *devPtr) override;
  void freeUnifiedMem(void *devPtr) override;
  void freePinnedMem(void *devPtr) override;
  std::string getMemLeaksReport() override;

  void *allocMemAsync(size_t size, void* streamPtr) override;
  void freeMemAsync(void *devPtr, void* streamPtr) override;

  void pinMemory(void* ptr, size_t size) override;
  void unpinMemory(void* ptr) override;
  void* devicePointer(void* ptr) override;

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

  bool isCapableOfGraphCapturing() override;
  DeviceGraphHandle streamBeginCapture(std::vector<void*>& streamPtrs) override;
  void streamEndCapture(DeviceGraphHandle handle) override;
  void launchGraph(DeviceGraphHandle graphHandle, void* streamPtr) override;

  void* createStream(double priority) override;
  void destroyGenericStream(void* streamPtr) override;
  void syncStreamWithHost(void* streamPtr) override;
  bool isStreamWorkDone(void* streamPtr) override;
  void syncStreamWithEvent(void* streamPtr, void* eventPtr) override;
  void streamHostFunction(void* streamPtr, const std::function<void()>& function) override;

  void streamWaitMemory(void* streamPtr, uint32_t* location, uint32_t value) override;

  void* createEvent() override;
  void destroyEvent(void* eventPtr) override;
  void syncEventWithHost(void* eventPtr) override;
  bool isEventCompleted(void* eventPtr) override;
  void recordEventOnHost(void* eventPtr) override;
  void recordEventOnStream(void* eventPtr, void* streamPtr) override;

  void initialize() override;
  void finalize() override;
  void profilingMessage(const std::string& message) override;
  void putProfilingMark(const std::string &name, ProfilingColors color) override;
  void popLastProfilingMark() override;

  bool isUnifiedMemoryDefault() override;

  void setupPrinting(int rank) override;

private:
  void createCircularStreamAndEvents();

  device::StatusT status{false};

  std::vector<cudaDeviceProp> properties;

  bool allowedConcurrentManagedAccess{false};
  
  bool usmDefault{false};
  bool canCompress{false};

  cudaStream_t defaultStream{nullptr};

  std::unordered_set<cudaStream_t> genericStreams{};

  struct GraphDetails {
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    std::vector<void*> streamPtrs;
    bool ready{false};
  };
  std::vector<GraphDetails> graphs;

  Statistics statistics{};
  std::unordered_map<void *, size_t> memToSizeMap{{nullptr, 0}};

  int priorityMin, priorityMax;
  InfoPrinter printer;

  std::unordered_map<void *, void *> allocationProperties;
};
} // namespace device


#endif // SEISSOLDEVICE_INTERFACES_CUDA_CUDAWRAPPEDAPI_H_

