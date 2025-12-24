#!/usr/bin/env sh

# SPDX-FileCopyrightText: 2021 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

export PREFERRED_DEVICE_TYPE=FPGA
source /opt/intel/inteloneapi/setvars.sh
mkdir -p build
cd build
cmake .. -DDEVICE_BACKEND=oneapi -DREAL_SIZE_IN_BYTES=4 -DDEVICE_ARCH=sm60 -DWITH_MPI=OFF -DFORCE_INTEL_MPI=ON
cmake --build .
