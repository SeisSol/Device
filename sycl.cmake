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
    set(HIPSYCL_TARGETS 'cuda:${DEVICE_SUB_ARCH}')
    find_package(hipSYCL CONFIG REQUIRED)
    add_sycl_to_target(TARGET device SOURCES ${DEVICE_SOURCE_FILES})
else()

    if("$ENV{ONEAPI_COMPILER}" STREQUAL "CLANG")
        set(CMAKE_CXX_COMPILER clang++)
    else()
        set(CMAKE_CXX_COMPILER dpcpp)
    endif()

    if("$ENV{PREFERRED_DEVICE_TYPE}" STREQUAL "FPGA")
        message(NOTICE "FPGA is used as target device, compilation will take several hours to complete!")
        target_compile_options(device PRIVATE "-fsycl" "-fintelfpga" "-fsycl-unnamed-lambda")
        set_target_properties(device PROPERTIES LINK_FLAGS "-fsycl -fintelfpga -Xshardware")
    elseif("$ENV{PREFERRED_DEVICE_TYPE}" STREQUAL "GPU")

        if(${DEVICE_SUB_ARCH} MATCHES "sm_*")
            if(NOT ("$ENV{ONEAPI_COMPILER}" STREQUAL "CLANG"))
                message(FATAL_ERROR "CUDA compilation only with CLANG compiler")
            endif()

            target_compile_options(device PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice" "-fsycl-unnamed-lambda" "-Xsycl-target-backend" "--cuda-gpu-arch=${DEVICE_SUB_ARCH}")
            set_target_properties(device PROPERTIES LINK_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Xs \"-device ${DEVICE_SUB_ARCH}\"")
        else()
            target_compile_options(device PRIVATE  "-fsycl" "-fsycl-targets=spir64_gen-unknown-unknown-sycldevice")
            set_target_properties(device PROPERTIES LINK_FLAGS "-fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs \"-device ${DEVICE_SUB_ARCH}\"")
        endif()
    elseif("$ENV{PREFERRED_DEVICE_TYPE}" STREQUAL "CPU")
        target_compile_options(device PRIVATE "-fsycl" "-fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice" "-fsycl-unnamed-lambda")
        set_target_properties(device PROPERTIES LINK_FLAGS "-fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs \"-march=${DEVICE_SUB_ARCH}\"")
    else()
        target_compile_options(device PRIVATE "-fsycl" "-fsycl-unnamed-lambda")
        message(WARNING "No device type specified for compilation, AOT and other platform specific details may be disabled")
    endif()
endif()

target_compile_options(device PRIVATE  "-O3")
target_link_libraries(device PUBLIC stdc++fs)
