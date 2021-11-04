#include "utils/logger.h"
#include "hip/hip_runtime.h"
#include "HipWrappedAPI.h"
#include "Internals.h"
#include <algorithm>

using namespace device;

void ConcreteAPI::copyTo(void* dst, const void* src, size_t count) {
  isFlagSet<DeviceSelected>(status);
  hipMemcpy(dst, src, count, hipMemcpyHostToDevice); CHECK_ERR;
  statistics.explicitlyTransferredDataToDeviceBytes += count;
}


void ConcreteAPI::copyFrom(void* dst, const void* src, size_t count) {
  isFlagSet<DeviceSelected>(status);
  hipMemcpy(dst, src, count, hipMemcpyDeviceToHost); CHECK_ERR;
  statistics.explicitlyTransferredDataToHostBytes += count;
}


void ConcreteAPI::copyBetween(void* dst, const void* src, size_t count) {
  isFlagSet<DeviceSelected>(status);
  hipMemcpy(dst, src, count, hipMemcpyDeviceToDevice); CHECK_ERR;
}


void ConcreteAPI::copyToAsync(void *dst, const void *src, size_t count, void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  auto stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipMemcpyAsync(dst, src, count, hipMemcpyHostToDevice, stream);
  CHECK_ERR;
}

void ConcreteAPI::copyFromAsync(void *dst, const void *src, size_t count, void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  auto stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToHost, stream);
  CHECK_ERR;
}

void ConcreteAPI::copyBetweenAsync(void *dst, const void *src, size_t count, void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  auto stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToDevice, stream);
  CHECK_ERR;
}


void ConcreteAPI::copy2dArrayTo(void *dst,
                                size_t dpitch,
                                const void *src,
                                size_t spitch,
                                size_t width,
                                size_t height) {
  isFlagSet<DeviceSelected>(status);
  hipMemcpy2D(dst, dpitch, src, spitch, width, height, hipMemcpyHostToDevice); CHECK_ERR;
  statistics.explicitlyTransferredDataToDeviceBytes += width * height;
}


void ConcreteAPI::copy2dArrayFrom(void *dst,
                                  size_t dpitch,
                                  const void *src,
                                  size_t spitch,
                                  size_t width,
                                  size_t height) {
  isFlagSet<DeviceSelected>(status);
  hipMemcpy2D(dst, dpitch, src, spitch, width, height, hipMemcpyDeviceToHost); CHECK_ERR;
  statistics.explicitlyTransferredDataToHostBytes += width * height;
}


void ConcreteAPI::prefetchUnifiedMemTo(Destination type, const void* devPtr, size_t count, void* streamPtr) {
  isFlagSet<InterfaceInitialized>(status);
  hipStream_t stream = (streamPtr == nullptr) ? 0 : (static_cast<hipStream_t>(streamPtr));
#ifndef NDEBUG
  auto itr = std::find(circularStreamBuffer.begin(), circularStreamBuffer.end(), streamPtr);
  if (itr == circularStreamBuffer.end()) {
    logError() << "DEVICE::ERROR: passed stream does not belong to circular stream buffer";
  }
#endif
  //hipMemPrefetchAsync - Not supported by HIP
  if(type == Destination::CurrentDevice){
    hipMemcpyAsync((void*) devPtr, devPtr, count, hipMemcpyHostToDevice, stream);
  }else{
    hipMemcpyAsync((void*) devPtr, devPtr, count, hipMemcpyDeviceToHost, stream);
  }
}
