#ifndef DEVICE_H
#define DEVICE_H

#include "AbstractAPI.h"
#include "Plasticity.h"

namespace device {

  enum MATRIX_LAYOUT{RowMajor, ColMajor};
  enum MATRIX_TRANSPOSE {NoTrans, Trans};

  // DeviceInstance -> Singleton
  class DeviceInstance {
  public:
    DeviceInstance(const DeviceInstance&) = delete;
    DeviceInstance &operator=(const DeviceInstance&) = delete;

    static DeviceInstance& getInstance() {
      static DeviceInstance Instance;
      return Instance;
    }

    template <typename AT, typename BT>
    void copyAddScale(const int m, const int n,
                      const real Alpha, AT A, const int lda,
                      const real Beta, BT B, const int ldb,
                      unsigned Offsets_A,
                      unsigned Offsets_B,
                      const unsigned NumElements);

    template <typename AT, typename BT, typename CT>
    void gemm(const MATRIX_LAYOUT Layout,
              const MATRIX_TRANSPOSE Transa,
              const MATRIX_TRANSPOSE Transb,
              const int m, const int n, const int k,
              const real Alpha, AT A_base, const int lda,
              BT B_base, const int ldb,
              const real Beta, CT C_base, const int ldc,
              unsigned Offsets_A,
              unsigned Offsets_B,
              unsigned Offsets_C,
              const unsigned NumElements = 1);

    ~DeviceInstance() {
      this->finalize();
    }

    void finalize();

    AbstractAPI *api = nullptr;
#ifdef USE_PLASTICITY
    device::Plasticity PlasticityLaunchers;
#endif

  private:
    DeviceInstance();
    unsigned m_MaxBlockSize{};
    unsigned m_MaxSharedMemSize{};
  };
}
#endif  //SEISSOL_DEVICE_H