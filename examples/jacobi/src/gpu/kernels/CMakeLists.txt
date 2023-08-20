#Add the kernel as a lib
cmake_minimum_required(VERSION 3.5)
project(subroutinesKernel)

set(TARGET_NAME kernel)

if (${DEVICE_BACKEND} STREQUAL "hip")
    include(hip.cmake)
elseif(${DEVICE_BACKEND} STREQUAL "cuda")
    include(cuda.cmake)
elseif((${DEVICE_BACKEND} STREQUAL "oneapi") OR (${DEVICE_BACKEND} STREQUAL "opensycl"))
    include(sycl.cmake)
endif()
