#include "SyclWrappedAPI.h"
#include "utils/logger.h"

namespace device {

void ConcreteAPI::checkOffloading() {
  const size_t N = 10;
  const real MAGIC_NUMBER = 19041995.0f;

  real hostPtr[N];
  for (size_t i = 0; i < N; i++) {
    hostPtr[i] = i;
  }

  auto *devPtr = (real *)this->allocGlobMem(N * sizeof(real));
  this->copyTo(devPtr, hostPtr, N * sizeof(real));

  auto *q = ((cl::sycl::queue *)getDefaultStream());
  q->submit([&](handler &cgh) {
    cgh.parallel_for(nd_range<>{{N}, {1}}, [=](nd_item<> item) {
      int i = item.get_global_id(0);
      devPtr[i] = MAGIC_NUMBER;
    });
  });



  this->copyFrom(hostPtr, devPtr, N * sizeof(real));
  this->freeMem(devPtr);

  for (size_t i = 0; i < N; i++) {
    if (hostPtr[i] != MAGIC_NUMBER) {
      throw std::invalid_argument("Kernel call did not result in expected result!");
    }
  }
  logInfo() << "Test Kernel successfully launched\n";

  logInfo() << this->currentStatistics->getSummary();
}

} // namespace device