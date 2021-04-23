#include "interfaces/sycl/Internals.h"
#include "subroutinesGPU.h"

#include <CL/sycl.hpp>
#include <device.h>

using namespace device::internals;
using namespace cl::sycl;


void launch_multMatVec(const GpuMatrixDataT &matrix, const real *v, real *res, void* streamPtr) {
  auto rng = computeExecutionRange1D(128, matrix.info.localNumRows);

  auto queue = reinterpret_cast<cl::sycl::queue*>(streamPtr);
  queue->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      int i = item.get_global_id(0);
      if (i < matrix.info.localNumRows) {
        const size_t row = matrix.info.range.start + i;
        res[row] = 0.0f;

        for (int columnIdx = 0; columnIdx < matrix.info.maxNonZerosPerRow; columnIdx++) {
          const int linIndex = row + columnIdx * matrix.info.numRows;
          const int column = matrix.indices[linIndex];

          if (column != NIN) {
            res[row] += (matrix.data[linIndex] * v[column]);
          }
        }
      }
    });
  });
}


void launch_manipVectors(const RangeT &range,
                         const real *vec1,
                         const real *vec2,
                         real *res,
                         VectorManipOps op,
                         void* streamPtr) {
  cl::sycl::nd_range<> rng = computeExecutionRange1D(128, range.end - range.start);

  auto queue = reinterpret_cast<cl::sycl::queue*>(streamPtr);
  queue->submit([&](handler &cgh) {
    cgh.parallel_for(rng, [=](nd_item<> item) {
      const size_t localSize = range.end - range.start;
      if (item.get_global_id(0) < localSize) {
        const int linIndex = range.start + item.get_global_id(0);
        switch (op) {
        case VectorManipOps::Addition: {
          res[linIndex] = vec1[linIndex] + vec2[linIndex];
          break;
        }
        case VectorManipOps::Subtraction: {
          res[linIndex] = vec1[linIndex] - vec2[linIndex];
          break;
        }
        case VectorManipOps::Multiply: {
          res[linIndex] = vec1[linIndex] * vec2[linIndex];
          break;
        }
        }
      }
    });
  });
}
