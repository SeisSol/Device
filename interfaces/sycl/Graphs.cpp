#include "HipWrappedAPI.h"
#include "utils/logger.h"


void *ConcreteAPI::getGraphInstance() {
  const auto id = currentDeviceId;
  logWarning(id) << "this device does not support compute-graph capturing";
  return nullptr;
}

void ConcreteAPI::launchGraph(void *userGraphInstance) {
  logError() << "DEVICE::ERROR: this device does not support computing via graph capturing";
}


void ConcreteAPI::syncGraph(void *userGraphInstance) {
  logError() << "DEVICE::ERROR: this device does not support computing via graph capturing";
}