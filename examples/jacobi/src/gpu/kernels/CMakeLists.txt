#Add the kernel as a lib
cmake_minimum_required(VERSION 3.5)
project(subroutinesKernel)

set(TARGET_NAME kernel)

if (${DEVICE_BACKEND} STREQUAL "HIP")
    include(amd.cmake)
elseif(${DEVICE_BACKEND} STREQUAL "CUDA")
    include(nvidia.cmake)
elseif((${DEVICE_BACKEND} STREQUAL "ONEAPI") OR (${DEVICE_BACKEND} STREQUAL "HIPSYCL"))
    include(sycl.cmake)
endif()
