#!/bin/bash
export PREFERRED_DEVICE_TYPE=FPGA
source /opt/intel/inteloneapi/setvars.sh
mkdir -p build
cd build
cmake .. -DDEVICE_BACKEND=ONEAPI -DREAL_SIZE_IN_BYTES=4 -DDEVICE_SUB_ARCH=sm60 -DWITH_MPI=OFF -DFORCE_INTEL_MPI=ON
cmake --build .