include(FindPackageHandleStandardArgs)
include(CheckCXXSourceRuns)
include(CMakePushCheckState)

cmake_push_check_state()
include(CheckCXXSourceCompiles)


set(_OMPT_TEST_PROGRAM "
#include <omp.h>
int main(int argc, char* argv[]) {
  #pragma omp target teams parallel
  {

  }
  return 0;
}")

if (NOT TARGET omptarget::device_flags)
  add_library(omptarget::device_flags INTERFACE IMPORTED)
  add_library(omptarget::interface INTERFACE IMPORTED)

    if(${DEVICE_ARCH} MATCHES "sm_*")
        set(_OMPT_DEVICE_FLAGS -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=${DEVICE_ARCH})
        target_compile_options(omptarget::device_flags INTERFACE -std=c++17 ${_OMPT_DEVICE_FLAGS} -Xcuda-ptxas -v)
        target_link_options(omptarget::device_flags INTERFACE ${_OMPT_DEVICE_FLAGS})
    elseif(${DEVICE_ARCH} MATCHES "gfx*")
        set(_OMPT_DEVICE_FLAGS -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=${DEVICE_ARCH})
        target_compile_options(omptarget::device_flags INTERFACE -std=c++17 ${_OMPT_DEVICE_FLAGS})
        target_link_options(omptarget::device_flags INTERFACE ${_OMPT_DEVICE_FLAGS})
    else()
        set(_OMPT_DEVICE_FLAGS -fopenmp -fopenmp-targets=intel_gpu_${DEVICE_ARCH})
        target_compile_options(omptarget::device_flags INTERFACE -std=c++17 ${_OMPT_DEVICE_FLAGS})
        target_link_options(omptarget::device_flags INTERFACE ${_OMPT_DEVICE_FLAGS})
    endif()
    set(_OMPT_DEVICE_FLAGS "") # TODO: remove
  
  set(CMAKE_REQUIRED_FLAGS ${_OMPT_DEVICE_FLAGS}) 
  check_cxx_source_compiles("${_OMPT_TEST_PROGRAM}" _OMPT_TEST_RUN_RESTULS)
endif()  


find_package_handle_standard_args(OmpTargetFlags _OMPT_DEVICE_FLAGS
                                             _OMPT_TEST_RUN_RESTULS)
cmake_pop_check_state()
