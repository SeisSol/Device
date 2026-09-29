# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

add_library(device ${DEVICE_LIBTYPE} device.cpp
                 interfaces/host/Control.cpp
                 interfaces/host/Memory.cpp
                 interfaces/host/Streams.cpp
                 algorithms/host/ArrayManip.cpp
                 algorithms/host/BatchManip.cpp
                 algorithms/host/Debugging.cpp
                 algorithms/host/Reduction.cpp)

set_target_properties(device PROPERTIES POSITION_INDEPENDENT_CODE ON)
