#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "datatypes.hpp"
#include "helper.hpp"
#include "host/subroutines.hpp"

using ::testing::ElementsAreArray;

class VectorTests : public ::testing::Test {
protected:
  void SetUp() override {
    v1.resize(VSIZE, 0);
    v2.resize(VSIZE, 0);

    int chunk = VSIZE / ws.size;
    range.start = ws.rank * chunk;
    range.end = ws.isLast ? VSIZE : (ws.rank + 1) * chunk;
  }

  int VSIZE{10};
  VectorT v1{};
  VectorT v2{};
  RangeT range{};
  WorkSpaceT ws{MPI_COMM_WORLD};
};

TEST_F(VectorTests, AssemblerTest) {
  WorkSpaceT ws{MPI_COMM_WORLD};

  // each MPI proc. init its part
  for (int i = range.start; i != range.end; ++i) {
    v1[i] = i;
  }

  // prepare a reference vector
  VectorT ref(VSIZE, 0);
  for (int i = 0; i < ref.size(); ++i) {
    ref[i] = i;
  }

  //VectorAssembler(WorkSpaceT ws, int start, int end)
  VectorAssembler assembler(ws, range);

  assembler.assemble(v1.data(), v2.data());
  ASSERT_THAT(v2, ElementsAreArray(ref));
}


#ifdef USE_MPI
TEST_F(VectorTests, InfNorm) {
  std::vector<real> extremes{-10.5, 20.01, 19.3}; // extreme values for each MPI proc.
  v1[0] = extremes[0];
  v1[5] = extremes[1];
  v1[9] = extremes[2];


  real localNorm = host::getInfNorm(range, v1);

  real globalNorm{};
  MPI_Allreduce(&localNorm, &globalNorm, 1, MPI_CUSTOM_REAL, MPI_MAX, ws.comm);
  ASSERT_FLOAT_EQ(20.01, globalNorm);
}
#endif


TEST_F(VectorTests, AddAndSubtract) {
  {
    VectorT test(VSIZE, 0);
    VectorT tmp (VSIZE, 0);

    // init
    for (int i = range.start; i != range.end; ++i) {
      v1[i] = real(i);
      v2[i] = -1.0 * real(i);
    }

    host::manipVectors(range, v1, v2, test, [](real a, real b){return a + b;});

    VectorAssembler assembler(ws, range);
    assembler.assemble(test.data(), tmp.data());

    VectorT ref(VSIZE, 0);
    ASSERT_THAT(tmp, ElementsAreArray(ref));
  }
  {
    VectorT test(VSIZE, 0);
    VectorT tmp (VSIZE, 0);

    // init
    for (int i = range.start; i != range.end; ++i) {
      v1[i] = real(i);
      v2[i] = -1.0 * real(i);
    }

    host::manipVectors(range, v1, v2, test, [](real a, real b){return a * b;});

    VectorAssembler assembler(ws, range);
    assembler.assemble(test.data(), tmp.data());

    VectorT ref(VSIZE, 0);
    for (int i = 0; i != VSIZE; ++i) {
      ref[i] = - real(i * i);
    }
    ASSERT_THAT(tmp, ElementsAreArray(ref));
  }
}
