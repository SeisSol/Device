#include "SyclWrappedAPI.h"

#include <algorithm>
#include <cassert>

using namespace device;

void *ConcreteAPI::getDefaultStream() {
  return &(this->currentQueueBuffer->getDefaultQueue());
}

void ConcreteAPI::syncDefaultStreamWithHost() {
  auto& defaultQueue = this->currentQueueBuffer->getDefaultQueue();
  this->currentQueueBuffer->syncQueueWithHost(&defaultQueue);
}

void* ConcreteAPI::getNextCircularStream() {
  assert(isCircularStreamsForked && "use a circular stream must be used inside a forked region");
  return &(this->currentQueueBuffer->getNextQueue());
}

void ConcreteAPI::resetCircularStreamCounter() {
  this->currentQueueBuffer->resetIndex();
}

size_t ConcreteAPI::getCircularStreamSize() {
  return this->currentQueueBuffer->getCapacity();
}

void ConcreteAPI::syncStreamFromCircularBufferWithHost(void* userStream) {
  auto *queuePtr = static_cast<cl::sycl::queue *>(userStream);
  this->currentQueueBuffer->syncQueueWithHost(queuePtr);
}

void ConcreteAPI::syncCircularBuffersWithHost() {
  this->currentQueueBuffer->syncAllQueuesWithHost();
}

void ConcreteAPI::forkCircularStreamsFromDefault() {
  assert(!isCircularStreamsForked && "circular streams must be joined before forking");

  this->syncDefaultStreamWithHost();
  isCircularStreamsForked = true;
}

void ConcreteAPI::joinCircularStreamsToDefault() {
  assert(isCircularStreamsForked && "circular streams must be forked before joining");

  this->syncCircularBuffersWithHost();
  isCircularStreamsForked = false;
}

bool ConcreteAPI::isCircularStreamsJoinedWithDefault() {
  return !isCircularStreamsForked;
}
