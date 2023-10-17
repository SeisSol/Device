set(DEVICE_SOURCE_FILES device.cpp
        interfaces/sycl/Aux.cpp
        interfaces/sycl/Control.cpp
        interfaces/sycl/Copy.cpp
        interfaces/sycl/Memory.cpp
        interfaces/sycl/Streams.cpp
        interfaces/sycl/DeviceContext.cpp
        interfaces/sycl/DeviceStack.cpp
        interfaces/sycl/DeviceCircularQueueBuffer.cpp
        interfaces/sycl/DeviceType.cpp
        algorithms/sycl/ArrayManip.cpp
        algorithms/sycl/BatchManip.cpp
        algorithms/sycl/Debugging.cpp
        )

if (${DEVICE_BACKEND} STREQUAL "oneapi")
    list(APPEND DEVICE_SOURCE_FILES "algorithms/sycl/Reduction_ONEAPI.cpp")
else()
    list(APPEND DEVICE_SOURCE_FILES "algorithms/sycl/Reduction.cpp")
endif()

add_library(device SHARED ${DEVICE_SOURCE_FILES})

if (${DEVICE_BACKEND} STREQUAL "hipsycl")
    set(HIPSYCL_TARGETS "cuda:${DEVICE_ARCH}")
    find_package(hipSYCL CONFIG REQUIRED)
    find_package(OpenMP REQUIRED)
    target_compile_options(device PRIVATE -Wno-unknown-cuda-version ${OpenMP_CXX_FLAGS})
    add_sycl_to_target(TARGET device SOURCES ${DEVICE_SOURCE_FILES})
    target_compile_definitions(device PRIVATE HIPSYCL_UNDERHOOD)
else()
    find_package(DpcppFlags REQUIRED)
    target_link_libraries(device PRIVATE dpcpp::device_flags)
    target_compile_definitions(device PRIVATE ONEAPI_UNDERHOOD)
endif()
