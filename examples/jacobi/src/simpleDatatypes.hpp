#ifndef JSOLVER_SIMPLE_DATATYPES_HPP
#define JSOLVER_SIMPLE_DATATYPES_HPP

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

enum SystemType{
  OnHost,
  OnDevice,
};

#endif // JSOLVER_SIMPLE_DATATYPES_HPP