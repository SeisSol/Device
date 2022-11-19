#NOTE: We need the device API to call kernels when using oneAPI
if (TARGET device)
    message("target device has already been added!")
else()
    add_subdirectory(root)
endif()

set(KERNEL_SOURCE_FILES subroutinesONEAPI.cpp)
add_library(${TARGET_NAME} SHARED ${KERNEL_SOURCE_FILES})
target_link_libraries(${TARGET_NAME} PUBLIC stdc++fs)

target_compile_options(${TARGET_NAME} PRIVATE ${EXTRA_FLAGS} -Wall -Wpedantic -std=c++17 -O3)
target_compile_definitions(${TARGET_NAME} PRIVATE DEVICE_${DEVICE_BACKEND}_LANG REAL_SIZE=${REAL_SIZE_IN_BYTES})

target_include_directories(${TARGET_NAME} PUBLIC root)
target_link_libraries(${TARGET_NAME} PUBLIC device)

if (${DEVICE_BACKEND} STREQUAL "hipsycl")
    if(NOT DEFINED HIPSYCL_TARGETS)
        set(HIPSYCL_TARGETS "cuda:sm_${DEVICE_ARCH}")
    endif()

    find_package(hipSYCL CONFIG REQUIRED)
    add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES ${DEVICE_SOURCE_FILES})
else()
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/root/cmake" "${CMAKE_ROOT}/Modules")
    find_package(DpcppFlags REQUIRED)
    target_link_libraries(${TARGET_NAME} PRIVATE dpcpp::device_flags)
endif()
