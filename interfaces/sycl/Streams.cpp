#include "SyclWrappedAPI.h"

#include <algorithm>

using namespace device;

void *ConcreteAPI::getNextCircularStream() { return &(this->currentQueueBuffer->getNextQueue()); }

void ConcreteAPI::resetCircularStreamCounter() { this->currentQueueBuffer->resetIndex(); }

size_t ConcreteAPI::getCircularStreamSize() { return this->currentQueueBuffer->getCapacity(); }

void ConcreteAPI::syncStreamFromCircularBuffer(void *streamPtr) {
  auto *q = static_cast<cl::sycl::queue *>(streamPtr);
  this->currentQueueBuffer->syncQueueWithHost(q);
}

void ConcreteAPI::syncCircularBuffer() { this->currentQueueBuffer->syncAllQueuesWithHost(); }

void ConcreteAPI::fastStreamsSync() { this->currentQueueBuffer->fastSync(); }

void *ConcreteAPI::getDefaultStream() { return &(this->currentQueueBuffer->getDefaultQueue()); }
