#ifndef DEVICE_PLASTICITY_H
#define DEVICE_PLASTICITY_H

#include "DataTypes.h"
#include <Initializer/PlasticityDataType.h>

namespace device {

  struct Plasticity {
    void saveFirstModes(real **ModalStressTensors,
                        real *FirsModes,
                        unsigned NumNodesPerElement,
                        unsigned NumElements);

    void adjustDeviatoricTensors(real **NodalStressTensors,
                                 PlasticityData *Plasticity,
                                 real *MeanStresses,
                                 real *Invariants,
                                 real *YieldFactor,
                                 unsigned *AdjustFlags,
                                 double RelaxTime,
                                 unsigned NumNodesPerElement,
                                 unsigned NumElements);


    void adjustModalStresses(unsigned* AdjustFlags,
                             real** NodalStressTensors,
                             real** ModalStressTensors,
                             real const* InverseVandermondeMatrix,
                             real* YieldFactor,
                             real* MeanStresses,
                             unsigned NumNodesPerElement,
                             unsigned NumElements);

    void computePstrains(unsigned* AdjustFlags,
                         real** ModalStressTensors,
                         real* FirsModes,
                         PlasticityData* Plasticity,
                         real (*Pstrains)[7],
                         double TimeStepWidth,
                         unsigned NumNodesPerElement,
                         unsigned NumElements);

    unsigned computeNumAdjustedDofs(unsigned* AdjustFlags,
                                    unsigned NumElements);
  };
}

#endif //DEVICE_PLASTICITY_H
