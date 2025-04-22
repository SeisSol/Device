// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_SYCL_SYCLWRAPPEDAPI_H_
#define SEISSOLDEVICE_INTERFACES_SYCL_SYCLWRAPPEDAPI_H_

#include "AbstractAPI.h"
#include "Common.h"
#include "DeviceContext.h"
#include "Statistics.h"

#include <CL/sycl.hpp>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

#if defined(DEVICE_USE_GRAPH_CAPTURING) && defined(SYCL_EXT_ONEAPI_GRAPH)
#define DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
#endif

// FIXME: suboptimal, too preprocessor-heavy for now
#if defined(HIPSYCL_EXT_ENQUEUE_CUSTOM_OPERATION)
#define DEVICE_SYCL_SUPPORTS_DIRECT_OPERATION
#define DEVICE_SYCL_DIRECT_OPERATION_NAME hipSYCL_enqueue_custom_operation
#elif defined(ACPP_EXT_ENQUEUE_CUSTOM_OPERATION)
#define DEVICE_SYCL_SUPPORTS_DIRECT_OPERATION
#define DEVICE_SYCL_DIRECT_OPERATION_NAME adaptiveCpp_enqueue_custom_operation
#elif defined(SYCL_EXT_ACPP_ENQUEUE_CUSTOM_OPERATION)
#define DEVICE_SYCL_SUPPORTS_DIRECT_OPERATION
#define DEVICE_SYCL_DIRECT_OPERATION_NAME ext_acpp_enqueue_custom_operation
#elif defined(SYCL_EXT_ONEAPI_ENQUEUE_CUSTOM_OPERATION)
#define DEVICE_SYCL_SUPPORTS_DIRECT_OPERATION
#define DEVICE_SYCL_DIRECT_OPERATION_NAME ext_oneapi_enqueue_custom_operation
#endif

#ifdef DEVICE_SYCL_SUPPORTS_DIRECT_OPERATION
#define DEVICE_SYCL_EMPTY_OPERATION(handle) handle.DEVICE_SYCL_DIRECT_OPERATION_NAME([=](...) {});
#define DEVICE_SYCL_EMPTY_OPERATION_WITH_EVENT(handle, event)                                                          \
  handle.depends_on(event);                                                                                            \
  handle.DEVICE_SYCL_DIRECT_OPERATION_NAME([=](...) {});
#elif defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER) && !defined(DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT)
#define DEVICE_SYCL_EMPTY_OPERATION(handle) handle.ext_oneapi_barrier();
#define DEVICE_SYCL_EMPTY_OPERATION_WITH_EVENT(handle, event) handle.ext_oneapi_barrier({event});
#else
#define DEVICE_SYCL_EMPTY_OPERATION(handle) handle.single_task([=]() {});
#define DEVICE_SYCL_EMPTY_OPERATION_WITH_EVENT(handle, event)                                                          \
  handle.depends_on(event);                                                                                            \
  handle.single_task([=]() {});
#endif

namespace device {
class ConcreteAPI : public AbstractAPI {

public:
  ConcreteAPI()
      : availableDevices{std::vector<DeviceContext *>{}}, currentDefaultQueue{nullptr}, currentDeviceStack{nullptr},
        currentQueueBuffer{nullptr}, currentStatistics{nullptr}, currentMemoryToSizeMap{nullptr}, deviceInitialized{false},
        currentDeviceId{0}{
    this->initDevices();
  }

  void setDevice(int deviceId) override;

  int getDeviceId() override;
  int getNumDevices() override;
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
  void copy2dArrayTo(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) override;
  void copy2dArrayFrom(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) override;
  void prefetchUnifiedMemTo(Destination type, const void *devPtr, size_t count, void *streamPtr) override;

  void copyToAsync(void *dst, const void *src, size_t count, void* streamPtr) override;
  void copyFromAsync(void *dst, const void *src, size_t count, void* streamPtr) override;
  void copyBetweenAsync(void *dst, const void *src, size_t count, void* streamPtr) override;

  size_t getMaxAvailableMem() override;
  size_t getCurrentlyOccupiedMem() override;
  size_t getCurrentlyOccupiedUnifiedMem() override;

  void *getDefaultStream() override;
  void syncDefaultStreamWithHost() override;

  bool isCapableOfGraphCapturing() override;
  void streamBeginCapture(std::vector<void*>& streamPtrs) override;
  void streamEndCapture() override;
  DeviceGraphHandle getLastGraphHandle() override;
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
  std::vector<DeviceContext*> availableDevices;

  cl::sycl::queue *currentDefaultQueue;
  bool isCircularStreamsForked{false};

  DeviceStack *currentDeviceStack;
  DeviceCircularQueueBuffer *currentQueueBuffer;
  Statistics *currentStatistics;
  std::unordered_map<void *, size_t> *currentMemoryToSizeMap;

#ifdef DEVICE_USE_GRAPH_CAPTURING_ONEAPI_EXT
  struct GraphDetails {
    std::optional<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>> instance;
    sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::modifiable> graph;
    bool ready{false};
  };
#else
  struct GraphDetails {
    bool ready{false};
  };
#endif

  std::vector<GraphDetails> graphs;

  void freeMem(void *devPtr);

  void initDevices();

  std::string getCurrentDeviceInfoAsText();
  std::string getDeviceInfoAsTextInternal(cl::sycl::device& dev);

  bool deviceInitialized;
  int currentDeviceId;
  InfoPrinter printer;
};
} // namespace device


#endif // SEISSOLDEVICE_INTERFACES_SYCL_SYCLWRAPPEDAPI_H_

