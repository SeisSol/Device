#include "device.h"

#ifdef CUDA
#include "CudaWrappedAPI.h"
#elif DUMMY
#include "DummyInterface.h"
#else
#error "Unknown interface for the device wrapper"
#endif

using namespace device;

Device::Device() {
  // NOTE: all headers inside of macros define their unique ConcreteInterface.
  // Make sure to not include multiple different interfaces at the same time.
  // Only one interface is allowed per program because of issues of unique compilers, etc.
  api = new ConcreteAPI;
  api->initialize();

  m_MaxBlockSize = api->getMaxThreadBlockSize();
  m_MaxSharedMemSize = api->getMaxSharedMemSize();
}

void Device::finalize() {
  if (!(api->hasFinalized())) {
    api->finalize();
  }
}

// copyAddScale -> is located within a concrete interface subdirectory. Note, it is interface specific
// gemm         -> is located within a concrete interface subdirectory. Note, it is interface specific