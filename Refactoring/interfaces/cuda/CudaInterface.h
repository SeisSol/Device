#ifndef SEISSOL_CUDAINTERFACE_H
#define SEISSOL_CUDAINTERFACE_H

#include "AbstractInterface.h"
namespace device {
  class ConcreteInterface : public AbstractInterface {
  public:
    void setDevice(int DeviceId);
    int getNumDevices();
    std::string getDeviceInfoAsText(int DeviceId);
    void synchDevice();
    void checkOffloading();

    void* allocGlobMem(size_t Size);
    void* allocUnifiedMem(size_t Size);
    void* allocPinnedMem(size_t Size);
    char* getTempMemory();
    void freeMem(void *DevPtr);
    void freePinnedMem(void *DevPtr);
    void freeTempMemory();

    void copyTo(void* Dst, const void* Src, size_t Count);
    void copyFrom(void* Dst, const void* Src, size_t Count);
    void copyBetween(void* Dst, const void* Src, size_t Count);
    void copy2dArrayTo(void *Dst, size_t Dpitch, const void *Src, size_t Spitch, size_t Width, size_t Height);
    void copy2dArrayFrom(void *Dst, size_t Dpitch, const void *Src, size_t Spitch, size_t Width, size_t Height);
    void streamBatchedData(real **BaseSrcPtr, real **BaseDstPtr, unsigned ElementSize, unsigned NumElements);
    void accumulateBatchedData(real **BaseSrcPtr, real **BaseDstPtr, unsigned ElementSize, unsigned NumElements);

    void compareDataWithHost(const real *HostPtr,
                             const real *DevPtr,
                             const size_t NumElements,
                             const char *ArrayName = nullptr);
    void initialize();
    void finalize();
  private:
    cudaStream_t Stream;
  };
}


#endif //SEISSOL_CUDAINTERFACE_H
