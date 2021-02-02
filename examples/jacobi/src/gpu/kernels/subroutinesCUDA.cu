#include "subroutinesGPU.h"
#include <cuda.h>


size_t get1DGrid(size_t blockSize, size_t size) {
  return (size + blockSize - 1) / blockSize;
}

//--------------------------------------------------------------------------------------------------
__global__ void kernel_multMatVec(GpuMatrixDataT matrix, const real *v, real *res) {
  const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < matrix.info.localNumRows) {
    const size_t row = matrix.info.range.start + idx;
    res[row] = 0.0f;

    for (int columnIdx = 0; columnIdx < matrix.info.maxNonZerosPerRow; ++columnIdx) {
      const int linIndex = row + columnIdx * matrix.info.numRows;
      const int column = matrix.indices[linIndex];
      if (column != NIN) {
        res[row] += (matrix.data[linIndex] * v[column]);
      }
    }
  }
}

void launch_multMatVec(const GpuMatrixDataT &matrix, const real *v, real *res) {
  dim3 block(128, 1, 1);
  dim3 grid(get1DGrid(block.x, matrix.info.localNumRows), 1, 1);
  kernel_multMatVec<<<grid, block>>>(matrix, v, res);
}

__global__ void kernel_manipVectors(RangeT range, const real* vec1, const real* vec2, real* res, VectorManipOps op) {
  const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t localSize = range.end - range.start;
  if (idx < localSize) {
    const int linIndex = range.start + idx;
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
}

void launch_manipVectors(const RangeT &range, const real* vec1, const real* vec2, real* res, VectorManipOps op) {
  size_t localSize = range.end - range.start;
  dim3 block(128, 1, 1);
  dim3 grid(get1DGrid(block.x, localSize), 1, 1);
  kernel_manipVectors<<<grid, block>>>(range, vec1, vec2, res, op);
}
