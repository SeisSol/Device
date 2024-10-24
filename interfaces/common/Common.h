#ifndef DEVICE_INTERFACE_STATUS_H
#define DEVICE_INTERFACE_STATUS_H

#include "utils/env.h"
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <cassert>

namespace device {
enum StatusID {
  DriverApiInitialized = 0,
  DeviceSelected,
  InterfaceInitialized,
  CircularStreamBufferInitialized,
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
}

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

inline size_t getMaxConcurrencyLevel(int defaultValue) {
  auto concurrencyLevel = utils::Env::get("DEVICE_MAX_CONCURRENCY_LEVEL",
                                          defaultValue);
  if (concurrencyLevel <= 0) {
    concurrencyLevel = defaultValue;
  }

  return static_cast<size_t>(concurrencyLevel);
}

constexpr auto mapPercentage(int minval, int maxval, double value) {
  const auto convminval = static_cast<double>(minval);
  const auto convmaxval = static_cast<double>(maxval);

  const auto base = (convmaxval + convminval) / 2;
  const auto offset = (convmaxval - convminval) / 2;

  const auto transformed = value * offset + base;

  return static_cast<int>(std::round(transformed));
}

#endif // DEVICE_INTERFACE_STATUS_H
