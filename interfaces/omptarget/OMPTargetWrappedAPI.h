#ifndef DEVICE_OMPTARGETINTERFACE_H
#define DEVICE_OMPTARGETINTERFACE_H

#include "AbstractAPI.h"
// #include "Statistics.h"
// #include "Common.h"

#include <stack>
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <cassert>

#include <omp.h>

namespace device {
class ConcreteAPI : public AbstractAPI {
public:
  ConcreteAPI() {
    circularStreamBuffer.resize(1000);
    defaultStream = new int[1];
    streamBuf = new int[1000];
    for (int i = 0; i < 1000; ++i) {
        circularStreamBuffer[i] = streamBuf + i;
    }
  }
  void setDevice(int deviceId) override {
    omp_set_default_device(deviceId);
  }

  int getDeviceId() override {
    return omp_get_default_device();
  }
  size_t getLaneSize() override {
    return 32; //?
  }
  int getNumDevices() override {
    return omp_get_num_devices();
  }
  unsigned getMaxThreadBlockSize() override {
    return omp_get_teams_thread_limit();
  }
  unsigned getMaxSharedMemSize() override {
    return 100; //?
  }
  unsigned getGlobMemAlignment() override {
    return 32; //?
  }
  std::string getDeviceInfoAsText(int deviceId) override {
    return "OMP ( ??? )";
  }
  void syncDevice() override {
    #pragma omp taskwait
  }
  void checkOffloading() override {
    // (none)
  }

  void allocateStackMem() override {
    stackMemory = reinterpret_cast<char*>(omp_target_alloc(maxStackMem, getDeviceId()));
  }
  void *allocGlobMem(size_t size) override {
    return omp_target_alloc(size, getDeviceId());
  }
  void *allocUnifiedMem(size_t size) override {
    return omp_alloc(size);
  }
  void *allocPinnedMem(size_t size) override {
    return omp_alloc(size);
  }
  char *getStackMemory(size_t requestedBytes) override {
    auto pos = stackMemByteCounter;
    stackMemByteCounter += requestedBytes;
    stackMemMeter.push(pos);
    return stackMemory + pos;
  }
  void freeMem(void *devPtr) override {
    omp_target_free(devPtr, getDeviceId());
  }
  void freePinnedMem(void *devPtr) override {
    omp_free(devPtr);
  }
  void popStackMemory() override {
    stackMemByteCounter -= stackMemMeter.top();
    stackMemMeter.pop();
  }
  std::string getMemLeaksReport() override {
    return "NYI";
  }

  std::string getApiName() override {
    return "OMP Target";
  }
  std::string getDeviceName(int deviceId) override {
    return "???";
  }

  void copyTo(void *dst, const void *src, size_t count) override {
    omp_target_memcpy(dst, src, count, 0, 0, omp_get_initial_device(), omp_get_default_device());
  }
  void copyFrom(void *dst, const void *src, size_t count) override {
    omp_target_memcpy(dst, src, count, 0, 0, omp_get_default_device(), omp_get_initial_device());
  }
  void copyBetween(void *dst, const void *src, size_t count) override {
    omp_target_memcpy(dst, src, count, 0, 0, omp_get_default_device(), omp_get_default_device());
  }
  void copyToAsync(void *dst, const void *src, size_t count, void* streamPtr) override {
    omp_depend_t depobj;
    int* stream = reinterpret_cast<int*>(streamPtr);
    #pragma omp depobj(depobj) depend(inout: stream[0])
    omp_target_memcpy_async(dst, src, count, 0, 0, omp_get_initial_device(), omp_get_default_device(), 1, &depobj);
    #pragma omp depobj(depobj) destroy
  }
  void copyFromAsync(void *dst, const void *src, size_t count, void* streamPtr) override {
    omp_depend_t depobj;
    int* stream = reinterpret_cast<int*>(streamPtr);
    #pragma omp depobj(depobj) depend(inout: stream[0])
    omp_target_memcpy_async(dst, src, count, 0, 0, omp_get_default_device(), omp_get_initial_device(), 1, &depobj);
    #pragma omp depobj(depobj) destroy
  }
  void copyBetweenAsync(void *dst, const void *src, size_t count, void* streamPtr) override {
    omp_depend_t depobj;
    int* stream = reinterpret_cast<int*>(streamPtr);
    #pragma omp depobj(depobj) depend(inout: stream[0])
    omp_target_memcpy_async(dst, src, count, 0, 0, omp_get_default_device(), omp_get_default_device(), 1, &depobj);
    #pragma omp depobj(depobj) destroy
  }



  void copy2dArrayTo(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                     size_t height) override {

                     }
  void copy2dArrayFrom(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                       size_t height) override {

                       }
  void prefetchUnifiedMemTo(Destination type, const void *devPtr, size_t count,
                            void *streamPtr) override {

                            }

  size_t getMaxAvailableMem() override {
    return 100000000000;
  }
  size_t getCurrentlyOccupiedMem() override {
    return 0;
  }
  size_t getCurrentlyOccupiedUnifiedMem() override {
    return 0;
  }

  void *getDefaultStream() override {
    return defaultStream;
  }
  void syncDefaultStreamWithHost() override {
    #pragma omp taskwait depend(inout: defaultStream[0])
  }

  void *getNextCircularStream() override {
    return reinterpret_cast<void*>(circularStreamBuffer[circularStreamCounter++]);
  }

  void resetCircularStreamCounter() override {
    circularStreamCounter = 0;
  }
  size_t getCircularStreamSize() override {
    return 1000;
  }
  void syncStreamFromCircularBufferWithHost(void* streamPtr) override {
    int* stream = reinterpret_cast<int*>(streamPtr);
    #pragma omp taskwait depend(inout: stream[0])
  }
  void syncCircularBuffersWithHost() override {
#pragma omp taskwait depend(inout: streamBuf[0:1000])
  }

  void forkCircularStreamsFromDefault() override {
    // for now, wait
    #pragma omp taskwait depend(inout: defaultStream[0])
  }
  void joinCircularStreamsToDefault() override {
    // for now, wait
    #pragma omp taskwait depend(inout: streamBuf[0:1000])
  }
  bool isCircularStreamsJoinedWithDefault() override {
    return true; //??
  }

  bool isCapableOfGraphCapturing() override {
    return false;
  }
  void streamBeginCapture() override{}
  void streamEndCapture() override{}
  DeviceGraphHandle getLastGraphHandle() override{return DeviceGraphHandle{};}
  void launchGraph(DeviceGraphHandle graphHandle) override{}
  void syncGraph(DeviceGraphHandle graphHandle) override{}

  void* createGenericStream() override {
    return new int[1];
  }
  void destroyGenericStream(void* streamPtr) override {
    int* stream = reinterpret_cast<int*>(streamPtr);
    delete[] stream;
  }
  void syncStreamWithHost(void* streamPtr) override {
    int* stream = reinterpret_cast<int*>(streamPtr);
    #pragma omp taskwait depend(inout: stream[0])
  }
  bool isStreamWorkDone(void* streamPtr) override {
    // for now, depend here
    int* stream = reinterpret_cast<int*>(streamPtr);
    #pragma omp taskwait depend(inout: stream[0])
    return true;
  }

  void initialize() override {}
  void finalize() override {}
  void putProfilingMark(const std::string &name, ProfilingColors color) override {}
  void popLastProfilingMark() override {}

private:
  void createCircularStreamAndEvents() {}

  // device::StatusT status{false};
  int currentDeviceId{-1};

  int* defaultStream{nullptr};
  int* streamBuf;

  std::vector<int*> circularStreamBuffer{};
  std::unordered_set<int*> genericStreams{};


  bool isCircularStreamsForked{false};
  size_t circularStreamCounter{0};

  char *stackMemory = nullptr;
  size_t stackMemByteCounter = 0;
  size_t maxStackMem = 1024 * 1024 * 1024; //!< 1GB in bytes
  std::stack<size_t> stackMemMeter{};

  // Statistics statistics{};
  std::unordered_map<void *, size_t> memToSizeMap{{nullptr, 0}};
};
} // namespace device

#endif // DEVICE_HIPINTERFACE_H
