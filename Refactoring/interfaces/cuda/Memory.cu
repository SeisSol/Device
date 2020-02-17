#include "CudaInterface.h"
#include "Internals.h"

using namespace device;


void* ConcreteInterface::allocGlobMem(size_t Size) {
  void *devPtr;
  cudaMalloc(&devPtr, Size); CHECK_ERR;
  return devPtr;
}


void* ConcreteInterface::allocUnifiedMem(size_t Size) {
  void *devPtr;
  cudaMallocManaged(&devPtr, Size, cudaMemAttachGlobal); CHECK_ERR;
  return devPtr;
}


void* ConcreteInterface::allocPinnedMem(size_t Size) {
  void *devPtr;
  cudaMallocHost(&devPtr, Size); CHECK_ERR;
  return devPtr;
}


void ConcreteInterface::freeMem(void *DevPtr) {
  cudaFree(DevPtr); CHECK_ERR;
}


void ConcreteInterface::freePinnedMem(void *DevPtr) {
  cudaFreeHost(DevPtr); CHECK_ERR;
}



char* ConcreteInterface::getTempMemory() {

}


void ConcreteInterface::freeTempMemory() {
  
}
