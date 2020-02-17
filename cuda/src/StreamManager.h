#ifndef SEISSOL_STREAMMANAGER_H
#define SEISSOL_STREAMMANAGER_H

#include <unordered_map>

namespace device {
  namespace streams {
    class Manager {
    public:
      static int getDefaultStream();
      int createStream();
      void destroyStream(int Id);
      void destroyAllStreams();

    private:
      std::unordered_map<int, void*> m_IdToStreamMap{};
      unsigned m_NumStreams{}
    };
  }
}


#endif //SEISSOL_STREAMMANAGER_H
