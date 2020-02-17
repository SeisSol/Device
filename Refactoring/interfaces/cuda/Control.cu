#include <iostream>
#include <string>
#include <sstream>
#include <cuda.h>

#include "CudaInterface.h"
#include "Internals.h"

using namespace device;

void ConcreteInterface::initialize() {
  cuInit(0); CHECK_ERR;
};


void ConcreteInterface::finalize() {};


void ConcreteInterface::setDevice(int DeviceId) {
  cudaSetDevice(DeviceId); CHECK_ERR;
}


int ConcreteInterface::getNumDevices() {
  int numDevices{};
  cudaGetDeviceCount(&numDevices); CHECK_ERR;
  return numDevices;
}


void ConcreteInterface::synchDevice() {
  cudaDeviceSynchronize(); CHECK_ERR;
}


std::string ConcreteInterface::getDeviceInfoAsText(int DeviceId) {
  cudaDeviceProp Property;
  cudaGetDeviceProperties(&Property, DeviceId); CHECK_ERR;

  std::ostringstream info;
  info << "Name: " << Property.name << '\n';
  info << "totalGlobalMem: " << Property.totalGlobalMem << '\n';
  info << "sharedMemPerBlock: " << Property.sharedMemPerBlock << '\n';
  info << "regsPerBlock: " << Property.regsPerBlock << '\n';
  info << "warpSize: " << Property.warpSize << '\n';
  info << "memPitch: " << Property.memPitch << '\n';
  info << "maxThreadsPerBlock: " << Property.maxThreadsPerBlock << '\n';
  info << "totalConstMem: " << Property.totalConstMem << '\n';
  info << "clockRate: " << Property.clockRate << '\n';
  info << "multiProcessorCount: " << Property.multiProcessorCount << '\n';
  info << "integrated: " << Property.integrated << '\n';
  info << "canMapHostMemory: " << Property.canMapHostMemory << '\n';
  info << "computeMode: " << Property.computeMode << '\n';
  info << "concurrentKernels: " << Property.concurrentKernels << '\n';
  info << "pciBusID: " << Property.pciBusID << '\n';
  info << "pciDeviceID: " << Property.pciDeviceID << '\n';

  return info.str();
}