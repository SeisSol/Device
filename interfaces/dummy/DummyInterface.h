#ifndef SEISSOL_DUMMYINTERFACE_H
#define SEISSOL_DUMMYINTERFACE_H

#include <AbstractInterface.h>
namespace device {
  class ConcreteInterface : public AbstractInterface {
  public:
    void setDevice(int DeviceId);
    int getNumDevices();
    std::string getDeviceInfoAsText(int DeviceId);
    void synchDevice();

    void* allocGlobMem(size_t Size);
    void* allocUnifiedMem(size_t Size);
    void* allocPinnedMem(size_t Size);
    void freeMem(void *DevPtr);
    void freePinnedMem(void *DevPtr);

    void copyTo(void* Dst, const void* Src, size_t Count);
    void copyFrom(void* Dst, const void* Src, size_t Count);
    void copyBetween(void* Dst, const void* Src, size_t Count);
    void copy2dArrayTo(void *Dst, size_t Dpitch, const void *Src, size_t Spitch, size_t Width, size_t Height);
    void copy2dArrayFrom(void *Dst, size_t Dpitch, const void *Src, size_t Spitch, size_t Width, size_t Height);

    void compareDataWithHost(const real *HostPtr,
                             const real *DevPtr,
                             const size_t NumElements,
                             const char *ArrayName = nullptr);
    void initialize();
    void finalize();
  private:
  };
}

#endif //SEISSOL_DUMMYINTERFACE_H
