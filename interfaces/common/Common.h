#ifndef DEVICE_INTERFACE_STATUS_H
#define DEVICE_INTERFACE_STATUS_H

#include "utils/env.h"
#include <array>
#include <cassert>
#include <cmath>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "utils/logger.h"

namespace device {
enum StatusID {
  DriverApiInitialized = 0,
  DeviceSelected,
  InterfaceInitialized,
  StackMemAllocated,
  Count
};

using StatusT = std::array<bool, StatusID::Count>;

template<StatusID ID>
void isFlagSet(const StatusT& status) { assert(status[ID]); };

template<typename T, typename U>
U align(T number, U alignment) {
  size_t alignmentFactor = (number + alignment - 1) / alignment;
  return alignmentFactor * alignment;
}
} // namespace device

inline int getMpiRankFromEnv() {
  std::vector<std::string> rankEnvVars{{"SLURM_PROCID"},
                                       {"OMPI_COMM_WORLD_RANK"},
                                       {"MV2_COMM_WORLD_RANK"},
                                       {"PMI_RANK"}};
  constexpr int defaultValue{-1};
  int value{defaultValue};
  utils::Env env;
  for (auto& envVar : rankEnvVars) {
    value = env.get(envVar.c_str(), defaultValue);
    if (value != defaultValue) break;
  }
  return (value != defaultValue) ? value : 0;
}

constexpr auto mapPercentage(int minval, int maxval, double value) {
  const auto convminval = static_cast<double>(minval);
  const auto convmaxval = static_cast<double>(maxval);

  const auto base = (convmaxval + convminval) / 2;
  const auto offset = (convmaxval - convminval) / 2;

  const auto transformed = value * offset + base;

  return static_cast<int>(std::round(transformed));
}

class InfoPrinter {
public:
  struct InfoPrinterLine {
    std::shared_ptr<std::ostringstream> stream;
    InfoPrinter& printer;

    InfoPrinterLine(InfoPrinter& printer) : printer(printer), stream(std::make_shared<std::ostringstream>()) {}

    template<typename T>
    InfoPrinterLine& operator <<(const T& data) {
      *stream << data;
      return *this;
    }

    ~InfoPrinterLine() {
      if (printer.rank < 0) {
        printer.stringCache.emplace_back(stream->str());
      }
      else {
        logInfo(printer.rank) << stream->str().c_str();
      }
      stream = nullptr;
    }
  };

  InfoPrinterLine printInfo() {
    return InfoPrinterLine(*this);
  }

  void setRank(int rank) {
    this->rank = rank;

    for (const auto& string : stringCache) {
      logInfo(rank) << string;
    }

    stringCache.resize(0);
  }

private:
  int rank{-1};
  std::vector<std::string> stringCache;
};

#endif // DEVICE_INTERFACE_STATUS_H
