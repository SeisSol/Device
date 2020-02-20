#include "StreamManager.h"
#include "common.h"

namespace device {
  namespace streams {

    int Manager::getDefaultStream() {
      return 0;
    }

    int Manager::createStream() {
      static uint64_t StreamCounter = 1;  //!< guarantees that the default stream is going to be unique

      cudaStream_t* pStream = new cudaStream_t;
      cudaStreamCreate(pStream); CUDA_CHECK;
      m_IdToStreamMap[StreamCounter] = static_cast<void*>(pStream);
      ++StreamCounter;
    }

    void Manager::destroyStream(int Id) {
      if (m_IdToStreamMap.find(Id) != m_IdToStreamMap.end()) {
        cudaStreamDestroy(*(static_cast<cudaStream_t*>(m_IdToStreamMap[Id]))); CUDA_CHECK;
        m_IdToStreamMap.erase(Id);
      }
    }

    void Manager::destroyAllStreams() {
      for (auto& Stream: m_IdToStreamMap) {
        cudaStreamDestroy(*(static_cast<cudaStream_t*>(Stream.second))); CUDA_CHECK;
      }
      m_IdToStreamMap.erase(m_IdToStreamMap.begin(), m_IdToStreamMap.end());
    }


  }
}