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

if (${DEVICE_BACKEND} STREQUAL "ONEAPI")
    list(APPEND DEVICE_SOURCE_FILES "algorithms/sycl/Reduction_ONEAPI.cpp")
else()
    list(APPEND DEVICE_SOURCE_FILES "algorithms/sycl/Reduction.cpp")
endif()

add_library(device SHARED ${DEVICE_SOURCE_FILES})

if (${DEVICE_BACKEND} STREQUAL "HIPSYCL")
    set(HIPSYCL_TARGETS)
    find_package(hipSYCL CONFIG REQUIRED)
    add_sycl_to_target(TARGET device SOURCES ${DEVICE_SOURCE_FILES})
else()
    set(CMAKE_CXX_COMPILER dpcpp)
    if("$ENV{PREFERRED_DEVICE_TYPE}" STREQUAL "FPGA")
        message(NOTICE "FPGA is used as target device, compilation will take several hours to complete!")
        target_compile_options(device PRIVATE "-fsycl" "-fintelfpga")
        set_target_properties(device PROPERTIES LINK_FLAGS "-fsycl -fintelfpga -Xshardware")
    elseif("$ENV{PREFERRED_DEVICE_TYPE}" STREQUAL "GPU")
        target_compile_options(device PRIVATE "-fsycl" "-fsycl-targets=spir64_gen-unknown-unknown-sycldevice")
        set_target_properties(device PROPERTIES LINK_FLAGS "-fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs \"-device ${DEVICE_SUB_ARCH}\"")
    elseif("$ENV{PREFERRED_DEVICE_TYPE}" STREQUAL "CPU")
        target_compile_options(device PRIVATE "-fsycl" "-fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice")
        set_target_properties(device PROPERTIES LINK_FLAGS "-fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs \"-march=${DEVICE_SUB_ARCH}\"")
    else()
        message(WARNING "No device type specified for compilation, AOT and other platform specific details may be disabled")
    endif()
endif()

target_compile_options(device PRIVATE "-std=c++17" "-O3")
target_link_libraries(device PUBLIC stdc++fs)
