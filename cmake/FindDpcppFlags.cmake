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
  
  if (NOT DEFINED DEVICE_ARCH)
    message(WARNING "DEVICE_ARCH is not set. "
                    "AOT and other platform specific details may be disabled")
    set(_USE_DEFAULT_COMPILATION ON)
  endif()

  if("$ENV{PREFERRED_DEVICE_TYPE}" STREQUAL "FPGA")
    message(NOTICE "FPGA is used as target device, compilation will take several hours to complete!")
    set(_DPCPP_DEVICE_FLAGS -fsycl -fintelfpga)
    target_compile_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS} -fsycl-unnamed-lambda)
    target_link_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS} -Xshardware)
  
  elseif("$ENV{PREFERRED_DEVICE_TYPE}" STREQUAL "GPU")
    if(${DEVICE_ARCH} MATCHES "sm_*")
      set(_DPCPP_DEVICE_FLAGS -fsycl
                              -Xsycl-target-backend
                              --cuda-gpu-arch=${DEVICE_ARCH}
                              -fsycl-targets=nvptx64-nvidia-cuda)
      target_compile_options(dpcpp::device_flags INTERFACE -std=c++17 ${_DPCPP_DEVICE_FLAGS})
      target_link_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS})
    else()
      set(_DPCPP_DEVICE_FLAGS -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice)
      target_compile_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS} -fsycl-unnamed-lambda)
      target_link_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS} -Xs -device ${DEVICE_ARCH})
    endif()
  
  elseif("$ENV{PREFERRED_DEVICE_TYPE}" STREQUAL "CPU")
    set(_DPCPP_DEVICE_FLAGS -fsycl -fsycl-targets=spir64_x86_64)
    target_compile_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS} -fsycl-unnamed-lambda)
    target_link_options(dpcpp::device_flags INTERFACE ${_DPCPP_DEVICE_FLAGS} -Xs \"-march=${DEVICE_ARCH}\")
  
  else()
    set(_USE_DEFAULT_COMPILATION ON)
    message(WARNING "No device type specified for compilation(env. PREFERRED_DEVICE_TYPE), "
                    "AOT and other platform specific details may be disabled")
  endif()

  if (_USE_DEFAULT_COMPILATION)
    target_compile_options(dpcpp::device_flags INTERFACE -fsycl -fsycl-unnamed-lambda)
    target_link_options(dpcpp::device_flags INTERFACE -fsycl)
  endif()
  
  set(CMAKE_REQUIRED_FLAGS "-fsycl") 
  check_cxx_source_compiles("${_SYCL_TEST_PROGRAM}" _SYCL_TEST_RUN_RESTULS)
endif()  


find_package_handle_standard_args(DpcppFlags _DPCPP_DEVICE_FLAGS
                                             _SYCL_TEST_RUN_RESTULS)
cmake_pop_check_state()
