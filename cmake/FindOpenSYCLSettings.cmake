include(FindPackageHandleStandardArgs)
include(CheckCXXSourceRuns)
include(CMakePushCheckState)

cmake_push_check_state()
include(CheckCXXSourceCompiles)

if (NOT TARGET acpp-settings::interface)
  if (NOT DEFINED DEVICE_ARCH)
    message(FATAL_ERROR "`DEVICE_ARCH` env. variable is not provided.")
  else()

  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_library(acpp-settings::interface INTERFACE IMPORTED)
 
  execute_process(COMMAND "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_LIST_DIR}/find-acpp.py -i
                  OUTPUT_VARIABLE _ACPP_INLCUDE_DIR)

  target_include_directories(acpp-settings::interface INTERFACE ${_ACPP_INLCUDE_DIR}
                                                                  ${_ACPP_INLCUDE_DIR}/sycl)

  execute_process(COMMAND "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_LIST_DIR}/find-acpp.py --vendor
                  OUTPUT_VARIABLE _ACPP_VENDOR)

  target_compile_definitions(acpp-settings::interface INTERFACE SYCL_PLATFORM_"${_ACPP_VENDOR}")


  execute_process(COMMAND "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_LIST_DIR}/find-acpp.py -t -a "${DEVICE_ARCH}"
                  OUTPUT_VARIABLE _ACPP_FULL_TARGET_NAME)
 
  set(HIPSYCL_TARGETS "${_ACPP_FULL_TARGET_NAME}")
  set(ACPP_TARGETS "${_ACPP_FULL_TARGET_NAME}")

  find_package(OpenMP REQUIRED)
  add_library(acpp-settings::device_flags INTERFACE IMPORTED)
  target_link_libraries(acpp-settings::device_flags INTERFACE OpenMP::OpenMP_CXX)
  set_property(TARGET acpp-settings::device_flags PROPERTY POSITION_INDEPENDENT_CODE ON)
    
  endif() 
endif()  


find_package_handle_standard_args(AdaptiveCppSettings _ACPP_INLCUDE_DIR
                                                   _ACPP_FULL_TARGET_NAME
                                                   _ACPP_VENDOR)
cmake_pop_check_state()
