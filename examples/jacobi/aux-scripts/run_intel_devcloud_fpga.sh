#!/usr/bin/env sh

# SPDX-FileCopyrightText: 2021 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

source /opt/intel/inteloneapi/setvars.sh
export PREFERRED_DEVICE_TYPE=FPGA
build/solver input/example.yaml
