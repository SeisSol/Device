include(FindPackageHandleStandardArgs)
include(CheckCXXSourceRuns)
include(CMakePushCheckState)

cmake_push_check_state()
include(CheckCXXSourceCompiles)


set(_SYCL_TEST_PROGRAM "
#include <CL/sycl.hpp>
int main(int argc, char* argv[]) {
  cl::sycl::queue queue;
  const unsigned size{32};
  auto *data = cl::sycl::malloc_shared<float>(size, queue);
  queue.submit([&] (cl::sycl::handler& cgh) {
    cgh.parallel_for(cl::sycl::range<1>(size),
                     [=](cl::sycl::id<1> i) {
      data[i] = cl::sycl::pown(static_cast<float>(i), 2);
    });
  });
  queue.wait();
  return 0;
}")

if (NOT TARGET dpcpp::device_flags)
  add_library(dpcpp::device_flags INTERFACE IMPORTED)
  add_library(dpcpp::interface INTERFACE IMPORTED)
  
  target_compile_definitions(dpcpp::device_flags INTERFACE __DPCPP_COMPILER)
  target_compile_definitions(dpcpp::interface INTERFACE __DPCPP_COMPILER)
  
  execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/find-dpcpp.py
                  OUTPUT_VARIABLE _DPCPP_ROOT)

  target_include_directories(dpcpp::interface INTERFACE ${_DPCPP_ROOT}/include
                                                        ${_DPCPP_ROOT}/include/sycl)

  set(DPCPP_PREFERRED_DEVICE_TYPE "GPU" CACHE STRING "The preferred device type to let DPC++ compile for")
  set(DPCPP_PREFERRED_DEVICE_TYPE_OPTIONS GPU CPU FPGA)
  set_property(CACHE DPCPP_PREFERRED_DEVICE_TYPE PROPERTY STRINGS ${DPCPP_PREFERRED_DEVICE_TYPE_OPTIONS})

  if(DEFINED ENV{PREFERRED_DEVICE_TYPE})
    set(DPCPP_PREFERRED_DEVICE_TYPE "$ENV{PREFERRED_DEVICE_TYPE}" CACHE STRING "" FORCE)
  endif()

  if("${DPCPP_PREFERRED_DEVICE_TYPE}" STREQUAL "FPGA")
    message(NOTICE "FPGA is used as target device, building may take a long time to complete.")
    set(_DPCPP_DEVICE_FLAGS -fsycl -fintelfpga)
    target_compile_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS} -fsycl-unnamed-lambda)
    target_link_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS} -Xshardware)
  
  elseif("${DPCPP_PREFERRED_DEVICE_TYPE}" STREQUAL "GPU")
    if(${DEVICE_ARCH} MATCHES "sm_*")
      set(_DPCPP_DEVICE_FLAGS -fsycl
                              -Xsycl-target-backend=nvptx64-nvidia-cuda
                              --cuda-gpu-arch=${DEVICE_ARCH}
                              -fsycl-targets=nvidia_gpu_${DEVICE_ARCH})
      target_compile_options(dpcpp::device_flags INTERFACE -std=c++17 ${_DPCPP_DEVICE_FLAGS} -fsycl-unnamed-lambda)
      target_link_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS})
    elseif(${DEVICE_ARCH} MATCHES "gfx*")
      set(_DPCPP_DEVICE_FLAGS -fsycl
                              -Xsycl-target-backend=amdgcn-amd-amdhsa
                              --offload-arch=${DEVICE_ARCH}
                              -fsycl-targets=amd_gpu_${DEVICE_ARCH})
      target_compile_options(dpcpp::device_flags INTERFACE -std=c++17 ${_DPCPP_DEVICE_FLAGS} -fsycl-unnamed-lambda)
      target_link_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS})
    else()
      set(_DPCPP_DEVICE_FLAGS -fsycl -fsycl-targets=intel_gpu_${DEVICE_ARCH})
      target_compile_options(dpcpp::device_flags INTERFACE -std=c++17 ${_DPCPP_DEVICE_FLAGS} -fsycl-unnamed-lambda)
      target_link_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS})
    endif()
  
  elseif("${DPCPP_PREFERRED_DEVICE_TYPE}" STREQUAL "CPU")
    set(_DPCPP_DEVICE_FLAGS -fsycl -fsycl-targets=spir64_x86_64)
    target_compile_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS} -fsycl-unnamed-lambda)
    target_link_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS} -Xs \"-march=${DEVICE_ARCH}\")
  
  else()
    target_compile_options(dpcpp::device_flags INTERFACE -fsycl -fsycl-unnamed-lambda)
    target_link_options(dpcpp::device_flags INTERFACE -fsycl)
    message(WARNING "No device type specified for the build (variable DPCPP_PREFERRED_DEVICE_TYPE), "
                    "AOT and other platform specific details may be disabled")
  endif()
  
  set(CMAKE_REQUIRED_FLAGS "-fsycl") 
  check_cxx_source_compiles("${_SYCL_TEST_PROGRAM}" _SYCL_TEST_RUN_RESTULS)
endif()  


find_package_handle_standard_args(DpcppFlags _DPCPP_DEVICE_FLAGS
                                             _SYCL_TEST_RUN_RESTULS)
cmake_pop_check_state()
