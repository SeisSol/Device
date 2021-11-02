#Add the kernel as a lib
cmake_minimum_required(VERSION 3.5)
project(subroutinesKernel)

set(TARGET_NAME kernel)

if (${DEVICE_BACKEND} STREQUAL "hip")
    include(amd.cmake)
elseif(${DEVICE_BACKEND} STREQUAL "cuda")
    include(nvidia.cmake)
elseif((${DEVICE_BACKEND} STREQUAL "oneapi") OR (${DEVICE_BACKEND} STREQUAL "hipsycl"))
    include(sycl.cmake)
endif()
