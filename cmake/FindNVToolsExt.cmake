# SPDX-FileCopyrightText: 2020-2024 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause


include(FindPackageHandleStandardArgs)

find_path(NVToolsExt_INCLUDE_DIRS nvToolsExt.h
          HINTS ${CUDA_TOOLKIT_ROOT_DIR}
          PATH_SUFFIXES include
          DOC "Directory where the nvToolsExt header is located")

find_library(NVToolsExt_LIBRARIES
             NAMES nvToolsExt
             HINTS ${CUDA_TOOLKIT_ROOT_DIR}
             PATH_SUFFIXES lib64 lib
             DOC "Directory where the nvToolsExt library is located")

find_package_handle_standard_args(NVToolsExt
                                  NVToolsExt_INCLUDE_DIRS
                                  NVToolsExt_LIBRARIES)

