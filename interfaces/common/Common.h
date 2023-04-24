#ifndef DEVICE_INTERFACE_STATUS_H
#define DEVICE_INTERFACE_STATUS_H

#include "utils/env.h"
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

#endif // DEVICE_INTERFACE_STATUS_H
