include(FindPackageHandleStandardArgs)
include(CheckCXXSourceRuns)
include(CMakePushCheckState)

cmake_push_check_state()
include(CheckCXXSourceCompiles)

if (NOT TARGET opensycl-settings::interface)
  if (NOT DEFINED DEVICE_ARCH)
    message(FATAL_ERROR "`DEVICE_ARCH` env. variable is not provided.")
  else()

  add_library(opensycl-settings::interface INTERFACE IMPORTED)
 
  execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/find-opensycl.py -i
                  OUTPUT_VARIABLE _OPENSYCL_INLCUDE_DIR)

  target_include_directories(opensycl-settings::interface INTERFACE ${_OPENSYCL_INLCUDE_DIR}
                                                                  ${_OPENSYCL_INLCUDE_DIR}/sycl)

  execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/find-opensycl.py --vendor
                  OUTPUT_VARIABLE _OPENSYCL_VENDOR)

  target_compile_definitions(opensycl-settings::interface INTERFACE SYCL_PLATFORM_"${_OPENSYCL_VENDOR}")


  execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/find-opensycl.py -t -a "${DEVICE_ARCH}"
                  OUTPUT_VARIABLE _OPENSYCL_FULL_TARGET_NAME)
 
  set(HIPSYCL_TARGETS "${_OPENSYCL_FULL_TARGET_NAME}")
  set(OPENSYCL_TARGETS "${_OPENSYCL_FULL_TARGET_NAME}")

  find_package(OpenMP REQUIRED)
  add_library(opensycl-settings::device_flags INTERFACE IMPORTED)
  target_link_libraries(opensycl-settings::device_flags INTERFACE OpenMP::OpenMP_CXX)
  set_property(TARGET opensycl-settings::device_flags PROPERTY POSITION_INDEPENDENT_CODE ON)
    
  endif() 
endif()  


find_package_handle_standard_args(OpenSYCLSettings _OPENSYCL_INLCUDE_DIR
                                                   _OPENSYCL_FULL_TARGET_NAME
                                                   _OPENSYCL_VENDOR)
cmake_pop_check_state()
