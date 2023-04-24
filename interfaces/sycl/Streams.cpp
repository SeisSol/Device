#include "SyclWrappedAPI.h"

#include <algorithm>
#include <cassert>

#ifdef ONEAPI_UNDERHOOD
#include <sycl/queue.hpp>
#endif // ONEAPI_UNDERHOOD

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


void* ConcreteAPI::createGenericStream() {
  // Note: in contrast to CUDA/HIP which can
  // create and handle an infinite number of stream,
  // SYCL support a limited number of streams.
  // This wrapper SYCL API creates and handles only
  // a single queue for generic stream operations
  // e.g., asynchronous data transfers
  return &(this->currentQueueBuffer->getGenericQueue());
}


void ConcreteAPI::destroyGenericStream(void*) {
  // no implementation is required
}


void ConcreteAPI::syncStreamWithHost(void* streamPtr) {
  auto *queuePtr = static_cast<cl::sycl::queue *>(streamPtr);
  this->currentQueueBuffer->syncQueueWithHost(queuePtr);
}


bool ConcreteAPI::isStreamWorkDone(void* streamPtr) {
  auto *queuePtr = static_cast<cl::sycl::queue *>(streamPtr);
#ifdef HIPSYCL_UNDERHOOD
  // Note: there is no way to check whether the queue
  // is empty or not in SYCL
  this->currentQueueBuffer->syncQueueWithHost(queuePtr);
  return true;
#endif // HIPSYCL_UNDERHOOD

#ifdef ONEAPI_UNDERHOOD
  // NOTE: use `ext_oneapi_empty` instead after
  // the next official release of dpc++
  this->currentQueueBuffer->syncQueueWithHost(queuePtr);
  return true;
#pragma message("switch to `ext_oneapi_empty` for the latest version of dpc++")
  //return queuePtr->ext_oneapi_empty();
#endif // ONEAPI_UNDERHOOD
}