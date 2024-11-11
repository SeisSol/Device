# SPDX-FileCopyrightText: 2024 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

include(FindPackageHandleStandardArgs)

set(MPI_CXX_COMPILER "mpiicpc")
set(MPI_CXX_LINK_FLAGS "-fsycl" "-lOpenCL" "-cxx=dpcpp")
set(MPI_CXX_COMPILE_OPTIONS "-cxx=dpcpp")

find_package_handle_standard_args(IntelMPI DEFAULT_MSG
        MPI_CXX_COMPILER
        MPI_CXX_LINK_FLAGS
        MPI_CXX_COMPILE_OPTIONS)

