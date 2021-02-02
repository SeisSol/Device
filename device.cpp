#include "device.h"

#ifdef DEVICE_CUDA_LANG
#include "CudaWrappedAPI.h"
#elif DEVICE_HIP_LANG
#include "HipWrappedAPI.h"
#else
#error "Unknown interface for the device wrapper"
#endif

using namespace device;

DeviceInstance::DeviceInstance() {
  // NOTE: all headers inside of macros define their unique ConcreteInterface.
  // Make sure to not include multiple different interfaces at the same time.
  // Only one interface is allowed per program because of issues of unique compilers, etc.
  api = new ConcreteAPI;
  api->initialize();
  algorithms.setDeviceApi(api);
}

void DeviceInstance::finalize() {
  if (api != nullptr) {
    api->finalize();
    delete api;
    api = nullptr;
  }
}
