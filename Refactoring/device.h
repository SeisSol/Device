#ifndef DEVICE_H
#define DEVICE_H

#include "AbstractAPI.h"

namespace device {

  enum MATRIX_LAYOUT{RowMajor, ColMajor};
  enum MATRIX_TRANSPOSE {NoTrans, Trans};

  // Device -> Singleton
  class Device {
  public:
    Device(const Device&) = delete;
    Device &operator=(const Device&) = delete;

    static Device& getInstance() {
      static Device Instance;
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

    ~Device() {
      this->finalize();
    }

    void finalize();

    AbstractAPI *api = nullptr;

  private:
    Device();
    unsigned m_MaxBlockSize{};
    unsigned m_MaxSharedMemSize{};
  };
}
#endif  //SEISSOL_DEVICE_H