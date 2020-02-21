#  NVToolsExt - NVIDIA Tools Extension Library Instantiation Software Framework
#
#  NOTE: it is not an official cmake search file
#  author: Ravil Dorozhinskii
#  email: ravil.dorozhinskii@tum.de
#
#  NVToolsExt_FOUND        - system has NVToolsExt
#  NVToolsExt_INCLUDE_DIRS - include directories for NVToolsExt
#  NVToolsExt_LIBRARIES    - libraries for NVToolsExt
#
#  Additional env. variables that may be used by this module.
#  They can change the default behaviour and
#  could to be set before calling find_package:
#
#  NVToolsExt_DIR          - the root directory of the NVToolsExt installation
#
# NOTE: One must call find_package(CUDA REQUIRED) before calling the file

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