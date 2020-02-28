#ifndef DEVICE_PLASTICITY_H
#define DEVICE_PLASTICITY_H

#include "DataTypes.h"
#include <Initializer/PlasticityDataType.h>

namespace device {

  struct Plasticity {
    void saveFirstModes(real *FirsModes,
                        const real **ModalStressTensors,
                        const unsigned NumNodesPerElement,
                        const unsigned NumElements);

    void adjustDeviatoricTensors(real **NodalStressTensors,
                                 unsigned *AdjustFlags,
                                 const PlasticityData *Plasticity,
                                 const double RelaxTime,
                                 const unsigned NumNodesPerElement,
                                 const unsigned NumElements);


    void adjustModalStresses(real** ModalStressTensors,
                             const real** NodalStressTensors,
                             const real* InverseVandermondeMatrix,
                             const unsigned* AdjustFlags,
                             const unsigned NumNodesPerElement,
                             const unsigned NumElements);

    void computePstrains(real **Pstrains,
                         const unsigned* AdjustFlags,
                         const real** ModalStressTensors,
                         const real* FirsModes,
                         const PlasticityData* Plasticity,
                         const double TimeStepWidth,
                         const unsigned NumNodesPerElement,
                         const unsigned NumElements);

    unsigned computeNumAdjustedDofs(unsigned* AdjustFlags,
                                    unsigned NumElements);
  };
}

#endif //DEVICE_PLASTICITY_H
