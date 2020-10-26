#ifndef DEVICE_PLASTICITY_H
#define DEVICE_PLASTICITY_H

#ifdef ACL_DEVICE

#include "DataTypes.h"
#include <Initializer/PlasticityDataType.h>

namespace device {

  struct Plasticity {
    void saveFirstModes(real *FirsModes,
                        const real **ModalStressTensors,
                        const unsigned NumNodesPerElement,
                        const unsigned NumElements);

    void adjustDeviatoricTensors(real **NodalStressTensors,
                                 int *Indices,
                                 const PlasticityData *Plasticity,
                                 const double RelaxTime,
                                 const unsigned NumNodesPerElement,
                                 const unsigned NumElements);

    unsigned getAdjustedIndices(int *Indices,
                                int *AdjustedIndices,
                                const unsigned NumElements);

    void adjustModalStresses(real** ModalStressTensors,
                             const real** NodalStressTensors,
                             const real* InverseVandermondeMatrix,
                             const int* AdjustedIndices,
                             const unsigned NumNodesPerElement,
                             const unsigned NumElements);

    void computePstrains(real **Pstrains,
                         const int* AdjustedIndices,
                         const real** ModalStressTensors,
                         const real* FirsModes,
                         const PlasticityData* Plasticity,
                         const double TimeStepWidth,
                         const unsigned NumNodesPerElement,
                         const unsigned NumElements);
  };
}

#endif //ACL_DEVICE

#endif //DEVICE_PLASTICITY_H
