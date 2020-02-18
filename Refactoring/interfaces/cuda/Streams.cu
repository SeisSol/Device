#include "assert.h"

#include "CudaInterface.h"
#include "Internals.h"
#include <iostream>

using namespace device;

unsigned ConcreteInterface::createStream(StreamType Type) {
  static size_t StreamIdCounter = 1;

  cudaStream_t* Stream = new cudaStream_t;
  cudaStreamCreateWithFlags(Stream,
                            Type == StreamType::Blocking ? cudaStreamDefault : cudaStreamNonBlocking); CHECK_ERR;
  unsigned StreamId = StreamIdCounter;

  assert((m_IdToStreamMap.find(StreamId) == m_IdToStreamMap.end())
          && (m_IdToStreamMap.find(0) == m_IdToStreamMap.end())
          && "DEVICE:: overflow w.r.t. the number of streams");

  m_IdToStreamMap[StreamId] = Stream;

  ++StreamIdCounter;
  return StreamId;
}


void ConcreteInterface::deleteStream(unsigned StreamId) {
  assert((m_IdToStreamMap.find(StreamId) != m_IdToStreamMap.end()) && "DEVICE: a stream doesn't exist");
  cudaStreamDestroy(*m_IdToStreamMap[StreamId]);
  m_IdToStreamMap.erase(StreamId);
}


void ConcreteInterface::deleteAllCreatedStreams() {
  for (auto& Stream: m_IdToStreamMap) {
    cudaStreamDestroy(*(Stream.second)); CHECK_ERR;
  }
  m_IdToStreamMap.erase(m_IdToStreamMap.begin(), m_IdToStreamMap.end());
  m_CurrentComputeStream = m_DefaultStream;
}


void ConcreteInterface::setComputeStream(unsigned StreamId) {
  assert((m_IdToStreamMap.find(StreamId) != m_IdToStreamMap.end()) && "DEVICE: a stream doesn't exist");
  m_CurrentComputeStream = *m_IdToStreamMap[StreamId];
}


void ConcreteInterface::setDefaultComputeStream() {
  m_CurrentComputeStream = m_DefaultStream;
}

__global__ void kernel_synchAllStreams() {
  // NOTE: an empty stream. It is supposed to get called with Cuda default stream. It is going to force all
  // other streams to finish their tasks
}


void ConcreteInterface::synchAllStreams() {
  kernel_synchAllStreams<<<1,1>>>();
}