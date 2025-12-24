// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_EXAMPLES_JACOBI_SRC_SIMPLEDATATYPES_HPP_
#define SEISSOLDEVICE_EXAMPLES_JACOBI_SRC_SIMPLEDATATYPES_HPP_

#if REAL_SIZE == 8
using real = double;
#else
using real = float;
#endif

#define NIN -1

struct RangeT {
  // Note we define a range in STL manner: [Start, End)
  int start{};
  int end{};
};

enum SystemType {
  OnHost,
  OnDevice,
};

#endif // SEISSOLDEVICE_EXAMPLES_JACOBI_SRC_SIMPLEDATATYPES_HPP_
