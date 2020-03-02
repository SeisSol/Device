#include <device_launch_parameters.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/distance.h>


#include "Plasticity.h"
#include "Internals.h"
#include "Gemm.cuh"


using namespace device;

#define NUM_STREESS_COMPONENTS 6

#if REAL_SIZE == 8
#define SQRT(X) sqrt(X)
#define MAX(X,Y) fmax(X,Y)
#elif REAL_SIZE == 4
#define SQRT(X) sqrtf(X)
#define MAX(X,Y) fmaxf(X,Y)
#else
#  error REAL_SIZE not supported.
#endif

__global__ void kernel_saveFirstMode(real *FirsModes,
                                     const real **ModalStressTensors,
                                     unsigned NumNodesPerElement) {
  FirsModes[threadIdx.x + blockDim.x * blockIdx.x] = ModalStressTensors[blockIdx.x][threadIdx.x * NumNodesPerElement];
}


void Plasticity::saveFirstModes(real *FirsModes,
                                const real **ModalStressTensors,
                                unsigned NumNodesPerElement,
                                unsigned NumElements) {
  dim3 Block(NUM_STREESS_COMPONENTS, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_saveFirstMode<<<Grid, Block>>>(FirsModes,
                                        ModalStressTensors,
                                        NumNodesPerElement);
  CHECK_ERR;
}


__global__ void kernel_adjustDeviatoricTensors(real **NodalStressTensors,
                                               int *Indices,
                                               const PlasticityData *Plasticity,
                                               const double RelaxTime) {
  __shared__ real InitialLoad[NUM_STREESS_COMPONENTS];
  real LocalStresses[NUM_STREESS_COMPONENTS];
  real *ElementTensors = NodalStressTensors[blockIdx.x];

  // 0. Load necessary data
  if (threadIdx.x < NUM_STREESS_COMPONENTS) {
    InitialLoad[threadIdx.x] = Plasticity[blockIdx.x].initialLoading[threadIdx.x];
  }
  __syncthreads();

  // 1. Add initial loading to the nodal stress tensor
  //#pragma unroll
  for (int i = 0; i < NUM_STREESS_COMPONENTS; ++i) {
    LocalStresses[i] = ElementTensors[threadIdx.x + blockDim.x * i] + InitialLoad[i];;
  }

  // 2. Compute the mean stress for each node
  real MeanStress = (LocalStresses[0] + LocalStresses[1] + LocalStresses[2]) / 3.0f;

  // 3. Compute deviatoric stress tensor
  //#pragma unroll
  for (int i = 0; i < 3; ++i) {
    LocalStresses[i] -= MeanStress;
  }

  // 4. Compute the second invariant for each node
  real Tau = 0.5 * (LocalStresses[0]*LocalStresses[0] + LocalStresses[1]*LocalStresses[1] + LocalStresses[2]*LocalStresses[2])
                 + (LocalStresses[3]*LocalStresses[3] + LocalStresses[4]*LocalStresses[4] + LocalStresses[5]*LocalStresses[5]);
  Tau = SQRT(Tau);

  // 5. Compute the plasticity criteria
  const real CohesionTimesCosAngularFriction = Plasticity[blockIdx.x].cohesionTimesCosAngularFriction;
  const real SinAngularFriction = Plasticity[blockIdx.x].sinAngularFriction;
  real Taulim = CohesionTimesCosAngularFriction - MeanStress * SinAngularFriction;
  Taulim = MAX(0.0, Taulim);

  __shared__ int Adjust;
  if (threadIdx.x == 0) {Adjust = -1;}

  // 6. Compute the yield factor
  real Factor = 0.0;
  if (Tau > Taulim) {
    Adjust = blockIdx.x;
    Factor = ((Taulim / Tau) - 1.0) * RelaxTime;
  }

  // 7. Adjust deviatoric stress tensor if a node within a node exceeds the elasticity region
  __syncthreads();
  if (Adjust != -1) {
    //#pragma unroll
    for (int i = 0; i < NUM_STREESS_COMPONENTS; ++i) {
      ElementTensors[threadIdx.x + blockDim.x * i] = LocalStresses[i] * Factor;
    }
  }

  if (threadIdx.x == 0) {
    Indices[blockIdx.x] = Adjust;
  }
  __syncthreads();
}

namespace device {
  struct MustGetCopied {
    __host__ __device__
    bool operator()(const int x) {
      return x != -1;
    }
  };
}

unsigned Plasticity::getAdjustedIndices(int *Indices, int *AdjustedIndices, const unsigned NumElements) {
  thrust::device_ptr<int> Begin(&Indices[0]);
  thrust::device_ptr<int> End(&Indices[NumElements]);
  thrust::device_ptr<int> ResultBegin(&AdjustedIndices[0]);

  auto ResultEnd = thrust::copy_if(thrust::device, Begin, End, ResultBegin, device::MustGetCopied()); CHECK_ERR;
  return thrust::distance(ResultBegin, ResultEnd);
}


void Plasticity::adjustDeviatoricTensors(real **NodalStressTensors,
                                         int *Indices,
                                         const PlasticityData *Plasticity,
                                         const double RelaxTime,
                                         const unsigned NumNodesPerElement,
                                         const unsigned NumElements) {
  dim3 Block(NumNodesPerElement, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_adjustDeviatoricTensors<<<Grid, Block>>>(NodalStressTensors,
                                                  Indices,
                                                  Plasticity,
                                                  RelaxTime);

  CHECK_ERR;
}


__global__ void kernel_adjustModalStresses(real** ModalStressTensors,
                                           const real** NodalStressTensors,
                                           const real* InverseVandermondeMatrix,
                                           const int* AdjustIndices,
                                           const unsigned NumNodesPerElement) {
    const int m = NumNodesPerElement;
    const int n = NUM_STREESS_COMPONENTS;
    const int k = NumNodesPerElement;

    dim3 Block(m, n, 1);
    size_t SharedMemSize = (m * k + k * n) * sizeof(real);

    kernel_gemmNN<<<1, Block, SharedMemSize>>>(m, n, k,
                                               1.0, InverseVandermondeMatrix, m,
                                               NodalStressTensors[AdjustIndices[blockIdx.x]], k,
                                               1.0, ModalStressTensors[AdjustIndices[blockIdx.x]], m,
                                               0, 0, 0);
}


void Plasticity::adjustModalStresses(real** ModalStressTensors,
                                     const real** NodalStressTensors,
                                     real const* InverseVandermondeMatrix,
                                     const int *AdjustIndices,
                                     const unsigned NumNodesPerElement,
                                     const unsigned NumElements) {
  dim3 Block(1, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_adjustModalStresses<<<Grid, Block>>>(ModalStressTensors,
                                              NodalStressTensors,
                                              InverseVandermondeMatrix,
                                              AdjustIndices,
                                              NumNodesPerElement);
  CHECK_ERR;
}


__global__ void kernel_computePstrains(real **Pstrains,
                                       const int* AdjustIndices,
                                       const real** ModalStressTensors,
                                       const real* FirsModes,
                                       const PlasticityData* Plasticity,
                                       const double TimeStepWidth,
                                       const unsigned NumNodesPerElement,
                                       const unsigned NumElements) {
  // NOTE: Six threads (x-dimension) work on the same element.
  size_t Index = threadIdx.y + blockIdx.x * blockDim.y;
  if (Index < NumElements) {
    // get local data
    real *LocalPstrains = Pstrains[AdjustIndices[Index]];
    const real *LocalModalTensor = ModalStressTensors[AdjustIndices[Index]];
    const real *LocalFirstMode = &FirsModes[NUM_STREESS_COMPONENTS * AdjustIndices[Index]];
    const PlasticityData *LocalData = &Plasticity[AdjustIndices[Index]];

    real DuDt_Pstrain = LocalData->mufactor * (LocalFirstMode[threadIdx.x]
                                               - LocalModalTensor[threadIdx.x * NumNodesPerElement]);
    LocalPstrains[threadIdx.x] += DuDt_Pstrain;

    __shared__ real Squared_DuDt_Pstrains[NUM_STREESS_COMPONENTS];
    real Factor = threadIdx.x < 3 ? 0.5f : 1.0f;
    Squared_DuDt_Pstrains[threadIdx.x] = Factor * DuDt_Pstrain * DuDt_Pstrain;
    __syncthreads();

    if (threadIdx.x == 0) {
      real Sum = 0.0;
      for (int i = 0; i < NUM_STREESS_COMPONENTS; ++i) {
        Sum += Squared_DuDt_Pstrains[i];
      }
      LocalPstrains[6] += (TimeStepWidth * SQRT(DuDt_Pstrain));
    }
  }
}

void Plasticity::computePstrains(real **Pstrains,
                                 const int* AdjustIndices,
                                 const real** ModalStressTensors,
                                 const real* FirsModes,
                                 const PlasticityData* Plasticity,
                                 const double TimeStepWidth,
                                 const unsigned NumNodesPerElement,
                                 const unsigned NumElements) {
  dim3 Block(NUM_STREESS_COMPONENTS, 32, 1);
  dim3 Grid = internals::computeGrid1D(Block.y, NumElements);
  kernel_computePstrains<<<Grid, Block>>>(Pstrains,
                                          AdjustIndices,
                                          ModalStressTensors,
                                          FirsModes,
                                          Plasticity,
                                          TimeStepWidth,
                                          NumNodesPerElement,
                                          NumElements);
  CHECK_ERR;
}



/*
__global__ void kernel_computePstrains(real *Pstrains,
                                       const real* ModalStressTensors,
                                       const real* FirsModes,
                                       const PlasticityData *Plasticity,
                                       const double TimeStepWidth,
                                       const unsigned NumNodesPerElement) {

  real DuDt_Pstrain = Plasticity->mufactor * (FirsModes[threadIdx.x]
                                              - ModalStressTensors[threadIdx.x * NumNodesPerElement]);
  Pstrains[threadIdx.x] += DuDt_Pstrain;

  __shared__ real Squared_DuDt_Pstrains[NUM_STREESS_COMPONENTS];
  real Factor = threadIdx.x < 3 ? 0.5f : 1.0f;
  Squared_DuDt_Pstrains[threadIdx.x] = Factor * DuDt_Pstrain * DuDt_Pstrain;
  __syncthreads();

  if (threadIdx.x == 0) {
    real Sum = 0.0;
    for (int i = 0; i < NUM_STREESS_COMPONENTS; ++i) {
      Sum += Squared_DuDt_Pstrains[i];
    }
    Pstrains[6] += (TimeStepWidth * SQRT(DuDt_Pstrain));
  }
}

__global__ void kernel_computePstrainsSelector(real **Pstrains,
                                               const int* AdjustIndices,
                                               const real** ModalStressTensors,
                                               const real* FirsModes,
                                               const PlasticityData* Plasticity,
                                               const double TimeStepWidth,
                                               const unsigned NumNodesPerElement) {

  int Index = AdjustIndices[blockIdx.x];
  kernel_computePstrains<<<1, NUM_STREESS_COMPONENTS>>>(Pstrains[Index],
                                                        ModalStressTensors[blockIdx.x],
                                                        &FirsModes[NUM_STREESS_COMPONENTS * Index],
                                                        &Plasticity[Index],
                                                        TimeStepWidth,
                                                        NumNodesPerElement);
}

void Plasticity::computePstrains(real **Pstrains,
                                 const int* AdjustIndices,
                                 const real** ModalStressTensors,
                                 const real* FirsModes,
                                 const PlasticityData* Plasticity,
                                 const double TimeStepWidth,
                                 const unsigned NumNodesPerElement,
                                 const unsigned NumElements) {
  dim3 Block(1, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_computePstrainsSelector<<<Grid, Block>>>(Pstrains,
                                                  AdjustIndices,
                                                  ModalStressTensors,
                                                  FirsModes,
                                                  Plasticity,
                                                  TimeStepWidth,
                                                  NumNodesPerElement);
  CHECK_ERR;
}
*/