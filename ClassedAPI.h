// SPDX-FileCopyrightText: 2019-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause




#ifndef SEISSOLDEVICE_CLASSEDAPI_H_
#define SEISSOLDEVICE_CLASSEDAPI_H_
#include <memory>
namespace device {

class DeviceEvent {
public:
  void* event() {
    return eventPtr.get();
  }
  void recordFromHost();
  void recordFromStream();
  void synchronize();
  int timestamp();
private:
  std::shared_ptr<void*> eventPtr;
};

class DeviceStream {
public:
  DeviceStream();
  ~DeviceStream();
  void* stream() {
    return streamPtr.get();
  }
  void synchronize();
  void dependsOnStream(DeviceStream stream) {
    dependsOnEvent(stream.event());
  }
  DeviceEvent event();
  void dependsOnEvent(DeviceEvent event);
  void hostFunction(const std::function<void()>& function) {

  }
private:
  std::shared_ptr<void*> streamPtr;
};

class DeviceStreamRingbuffer {
public:
  DeviceStreamRingbuffer(std::size_t size);
  void join(DeviceStream stream);
  void fork(DeviceStream stream);
  DeviceStream get(std::size_t id) {
    return streams[id % streams.size()];
  }
  DeviceStream next() {
    return get(position++);
  }
  template<typename F>
  void distribute(DeviceStream base, std::size_t count, F&& handler) {
    fork(base);
    for (std::size_t i = 0; i < count; ++i) {
      handler(i, get(i));
    }
    join(base);
  }

  std::vector<DeviceStream>& getStreams() {
    return streams;
  }
private:
  std::vector<DeviceStream> streams;
  std::size_t position;
};

class DeviceGraph {
public:
  void captureBegin(std::vector<DeviceStream>& streams);
  void captureStop();
  void replay(DeviceStream& stream);

  template<typename F>
  void run(std::vector<DeviceStream>& streams, DeviceStream& replayStream, F&& handler) {
    if (capturable) {
      if (!captured) {
        captureBegin(streams);
        std::invoke(std::forward<F>(handler), streams); 
        captureStop();
      }
      replay(replayStream);
    }
    else {
      std::invoke(std::forward<F>(handler), streams); 
    }
  }
private:
  bool captured;
};

} // namespace device



#endif // SEISSOLDEVICE_CLASSEDAPI_H_

