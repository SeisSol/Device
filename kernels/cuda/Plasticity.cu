#include <device_launch_parameters.h>
#include "Plasticity.h"
#include "Internals.h"
#include "Gemm.cuh"


using namespace device;

#define NUM_STREESS_COMPONENTS 6

#if REAL_SIZE == 8
#define SQRT(x) sqrtf(x)
#elif REAL_SIZE ==4
#define SQRT(x) sqrt(x)
#else
#  error REAL_SIZE not supported.
#endif

__global__ void kernel_saveFirstMode(real **ModalStressTensors,
                                     real *FirsModes,
                                     unsigned NumNodesPerElement) {
  const real FirstModeValue = ModalStressTensors[blockIdx.x][threadIdx.x * NumNodesPerElement];
  FirsModes[threadIdx.x + blockDim.x * blockIdx.x] = FirstModeValue;
}

#include <iostream>
void Plasticity::saveFirstModes(real **ModalStressTensors,
                                real *FirsModes,
                                unsigned NumNodesPerElement,
                                unsigned NumElements) {

  dim3 Block(6, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_saveFirstMode<<<Grid, Block>>>(ModalStressTensors,
                                            FirsModes,
                                            NumNodesPerElement);
  CHECK_ERR;
}


__global__ void kernel_adjustDeviatoricTensors(real **NodalStressTensors,
                                               PlasticityData *Plasticity,
                                               real *MeanStresses,
                                               real *Invariants,
                                               real *YieldFactor,
                                               unsigned *AdjustFlags,
                                               double RelaxTime) {
  __shared__ real InitialLoad[NUM_STREESS_COMPONENTS];
  real LocalStresses[NUM_STREESS_COMPONENTS];
  real *ElementTensors = NodalStressTensors[blockIdx.x];

  // 0. Load necessary data
  if (threadIdx.x < NUM_STREESS_COMPONENTS) {
    InitialLoad[threadIdx.x] = Plasticity[blockIdx.x].initialLoading[threadIdx.x];
  }
  __syncthreads();

  #pragma unroll
  for (int i = 0; i < NUM_STREESS_COMPONENTS; ++i) {
    LocalStresses[i] = ElementTensors[threadIdx.x + blockDim.x * i];
  }

  // 1. Add initial loading to the nodal stress tensor
  #pragma unroll
  for (int i = 0; i < NUM_STREESS_COMPONENTS; ++i) {
    LocalStresses[i] += InitialLoad[i];
  }

  // 2. Compute the mean stress for each node
  real MeanStress = (LocalStresses[0] + LocalStresses[1] + LocalStresses[2]) / 3.0f;
  MeanStresses[threadIdx.x + blockDim.x * blockIdx.x] = MeanStress;

  // 3. Compute deviatoric stress tensor
  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    LocalStresses[i] -= MeanStress;
  }

  // 4. Compute the second invariant for each node
  real Invariant = 0.0;

  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    Invariant += LocalStresses[i] * LocalStresses[i];
  }

  Invariant *= 0.5;

  #pragma unroll
  for (int i = 3; i < 6; ++i) {
    Invariant += LocalStresses[i] * LocalStresses[i];
  }
  Invariant = SQRT(Invariant);

  // 5. Compute the plasticity criteria
  const real CohesionTimesCosAngularFriction = Plasticity[blockIdx.x].cohesionTimesCosAngularFriction;
  const real SinAngularFriction = Plasticity[blockIdx.x].sinAngularFriction;

  real Taulim = CohesionTimesCosAngularFriction - MeanStress * SinAngularFriction;
  Taulim = Taulim > 0.0f ? Taulim : 0.0f;

  __shared__ unsigned Adjust;
  if (threadIdx.x == 0) {Adjust = 0;}

  // 6. Compute the yield factor
  real Factor = 0.0;
  if (Invariant > Taulim) {
    Adjust = 1;
    Factor = (Taulim / Invariant - 1.0) * RelaxTime;
  }

  // 7. Adjust deviatoric stress tensor if a node within a node exceeds the elasticity region
  __syncthreads();
  if (Adjust != 0) {
    #pragma unroll
    for (int i = 0; i < NUM_STREESS_COMPONENTS; ++i) {
      LocalStresses[i] *= Factor;
      ElementTensors[threadIdx.x + blockDim.x * i] = LocalStresses[i];
    }
  }

  if (threadIdx.x == 0) {
    AdjustFlags[blockIdx.x] = Adjust;
  }
}


void Plasticity::adjustDeviatoricTensors(real **NodalStressTensors,
                                         PlasticityData *Plasticity,
                                         real *MeanStresses,
                                         real *Invariants,
                                         real *YieldFactor,
                                         unsigned *AdjustFlags,
                                         double RelaxTime,
                                         unsigned NumNodesPerElement,
                                         unsigned NumElements) {
  dim3 Block(NumNodesPerElement, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_adjustDeviatoricTensors<<<Grid, Block>>>(NodalStressTensors,
                                                  Plasticity,
                                                  MeanStresses,
                                                  Invariants,
                                                  YieldFactor,
                                                  AdjustFlags,
                                                  RelaxTime);
  CHECK_ERR;
}


__global__ void kernel_adjustModalStresses(unsigned* AdjustFlags,
                                           real** NodalStressTensors,
                                           real** ModalStressTensors,
                                           real const* InverseVandermondeMatrix,
                                           real* YieldFactor,
                                           real* MeanStresses,
                                           unsigned NumNodesPerElement) {
  if (AdjustFlags[blockIdx.x] != 0) {
    const int m = NumNodesPerElement;
    const int n = NUM_STREESS_COMPONENTS;
    const int k = NumNodesPerElement;

    dim3 Block(m, n, 1);
    size_t SharedMemSize = (m * k + k * n) * sizeof(real);
    kernel_gemmNN<<<1, Block, SharedMemSize>>>(m, n, k,
                                               1.0, InverseVandermondeMatrix, m,
                                               NodalStressTensors[blockDim.x], k,
                                               1.0, ModalStressTensors[blockDim.x], m,
                                               0, 0, 0);
  }
}


void Plasticity::adjustModalStresses(unsigned *AdjustFlags,
                                     real** NodalStressTensors,
                                     real** ModalStressTensors,
                                     const real *InverseVandermondeMatrix,
                                     real *YieldFactor,
                                     real* MeanStresses,
                                     unsigned NumNodesPerElement,
                                     unsigned NumElements) {
  dim3 Block(1, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_adjustModalStresses<<<Grid, Block>>>(AdjustFlags,
                                              NodalStressTensors,
                                              ModalStressTensors,
                                              InverseVandermondeMatrix,
                                              YieldFactor,
                                              MeanStresses,
                                              NumNodesPerElement);
  CHECK_ERR;
}



__global__ void kernel_computePstrains(real* ModalStressTensors,
                                       real* FirsModes,
                                       PlasticityData *Plasticity,
                                       real Pstrains[7],
                                       double TimeStepWidth,
                                       unsigned NumNodesPerElement) {

  real DuDt_Pstrain = Plasticity->mufactor * (FirsModes[threadIdx.x]
                                              - ModalStressTensors[threadIdx.x * NumNodesPerElement]);
  Pstrains[threadIdx.x] += DuDt_Pstrain;

  __shared__ real Squared_DuDt_Pstrains[6];
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

__global__ void kernel_computePstrainsSelector(unsigned* AdjustFlags,
                                               real** ModalStressTensors,
                                               real* FirsModes,
                                               PlasticityData* Plasticity,
                                               real (*Pstrains)[7],
                                               double TimeStepWidth,
                                               unsigned NumNodesPerElement) {
  if (AdjustFlags[blockIdx.x] != 0) {
    kernel_computePstrains<<<1, 6>>>(ModalStressTensors[blockIdx.x],
                                     &FirsModes[NUM_STREESS_COMPONENTS * blockIdx.x],
                                     &Plasticity[blockIdx.x],
                                     Pstrains[blockIdx.x],
                                     TimeStepWidth,
                                     NumNodesPerElement);
  }
}

void Plasticity::computePstrains(unsigned* AdjustFlags,
                                 real** ModalStressTensors,
                                 real* FirsModes,
                                 PlasticityData* Plasticity,
                                 real (*Pstrains)[7],
                                 double TimeStepWidth,
                                 unsigned NumNodesPerElement,
                                 unsigned NumElements) {
  dim3 Block(1, 1, 1);
  dim3 Grid(NumElements, 1, 1);
  kernel_computePstrainsSelector<<<Grid, Block>>>(AdjustFlags,
                                                  ModalStressTensors,
                                                  FirsModes,
                                                  Plasticity,
                                                  Pstrains,
                                                  TimeStepWidth,
                                                  NumNodesPerElement);
  CHECK_ERR;
}

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
unsigned Plasticity::computeNumAdjustedDofs(unsigned* AdjustFlags,
                                            unsigned NumElements) {

  thrust::device_ptr<unsigned> Begin(&AdjustFlags[0]);
  thrust::device_ptr<unsigned> End(&AdjustFlags[NumElements]);
  return thrust::reduce(Begin, End, (unsigned) 0, thrust::plus<unsigned>());
}